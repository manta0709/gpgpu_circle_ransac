#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "curand.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <random>
#include <fstream>
#include <math.h>

#include <opencv2/opencv.hpp>

#pragma region Prev

struct Data
{
	cv::Mat img;
	std::vector<cv::Point2i> points;
};

constexpr int img_count = 4;
std::mt19937 rng;
std::uniform_real_distribution<float> udist;
const int iters = 100000;
const int threadNum = 256;
const float threshold = .05f;

cv::Mat normalize(std::vector<cv::Point2f>& pts)
{
	int ptsNum = pts.size();

	cv::Point2f mean(0, 0);

	for (int i = 0; i < ptsNum; i++) {
		mean += pts[i];
	}

	mean /= (float)ptsNum;

	float spread = 0.0;

	for (int i = 0; i < ptsNum; i++) {
		spread += cv::norm(pts[i] - mean);
	}

	spread /= (float)ptsNum;

	cv::Mat offs = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat scale = cv::Mat::eye(3, 3, CV_32F);

	offs.at<float>(0, 2) = -mean.x;
	offs.at<float>(1, 2) = -mean.y;

	scale.at<float>(0, 0) = sqrt(2) / spread;
	scale.at<float>(1, 1) = sqrt(2) / spread;

	cv::Mat T = scale * offs;

	for (int i = 0; i < ptsNum; i++) {
		pts[i].x = sqrt(2) * (pts[i].x - mean.x) / spread;
		pts[i].y = sqrt(2) * (pts[i].y - mean.y) / spread;
	}

	return T;
}

inline float distance(const cv::Point3f& c, const cv::Point2f& p)
{
	return abs(sqrtf(powf(p.x - c.x, 2) + powf(p.y - c.y, 2)) - c.z);
}

std::vector<cv::Point2f> filterInliers(const cv::Point3f& params, const std::vector<cv::Point2f>& pts, const float& threshold)
{
	std::vector<cv::Point2f> inlier_points;

	for (const cv::Point2f& p : pts)
	{
		if (distance(params, p) < threshold) {
			inlier_points.push_back(p);
		}
	}

	return inlier_points;
}

cv::Point3f getCircleParams(const std::vector<cv::Point2f>& points)
{
	cv::Mat A(points.size(), 3, CV_32F);
	cv::Mat B(points.size(), 1, CV_32F);

	for (size_t i = 0; i < points.size(); i++)
	{
		A.at<float>(i, 0) = points[i].x;
		A.at<float>(i, 1) = points[i].y;
		A.at<float>(i, 2) = 1.0f;

		B.at<float>(i, 0) = powf(points[i].x, 2) + powf(points[i].y, 2);
	}

	cv::Mat X(3, 1, CV_32F);
	cv::solve(A, B, X, cv::DECOMP_NORMAL);
	//gives [2xc, 2yc, r^2-xc^2-yc^2]
	float x = X.at<float>(0, 0) / 2.f;
	float y = X.at<float>(1, 0) / 2.f;
	float r = sqrtf(X.at<float>(2, 0) + powf(x, 2) + powf(y, 2));

	return { x, y, r };
}

std::vector<cv::Point2f> detectCircleCpu(const std::vector<cv::Point2f>& pts, const float& threshold, const int& iters)
{
	int mostInliers = 0;
	std::vector<cv::Point2f> inliers;

	int idx1, idx2, idx3;
	for (int i = 0; i < iters; i++)
	{
		do
		{
			idx1 = udist(rng) * pts.size() - 1;
			idx2 = udist(rng) * pts.size() - 1;
			idx3 = udist(rng) * pts.size() - 1;
		} while (idx1 == idx2 || idx1 == idx3 || idx2 == idx3);

		std::vector<cv::Point2f> tmp = filterInliers(getCircleParams({ pts[idx1], pts[idx2], pts[idx3] }), pts, threshold);

		if (tmp.size() > mostInliers)
		{
			mostInliers = tmp.size();
			inliers = tmp;
		}
	}

	return filterInliers(getCircleParams(inliers), pts, threshold);
}

struct parallelInlierStruct
{
	cv::Point3f c;
	int size;
};

std::vector<cv::Point2f> detectCircleCpuParallel(const std::vector<cv::Point2f>& pts, const float& threshold, const int& iters)
{
	parallelInlierStruct bestModel;
	bestModel.size = 0;
	std::vector<parallelInlierStruct> result;
	result.resize(iters);

	std::mutex v_lock;
	std::vector<std::thread> threads;
	std::mutex random_mutex;

	auto ransaciter = [&result, &v_lock, &random_mutex, &pts, &threshold](const int& id) {
		int idx1, idx2, idx3;

		std::unique_lock<std::mutex> r_lock(random_mutex);
		do
		{
			idx1 = udist(rng) * pts.size() - 1;
			idx2 = udist(rng) * pts.size() - 1;
			idx3 = udist(rng) * pts.size() - 1;
		} while (idx1 == idx2 || idx1 == idx3 || idx2 == idx3);
		r_lock.unlock();

		cv::Point3f params = getCircleParams({ pts[idx1], pts[idx2], pts[idx3] });

		parallelInlierStruct tmp;
		tmp.c = params;
		tmp.size = 0;

		for (const cv::Point2f& p : pts)
		{
			if (distance(params, p) < threshold) {
				tmp.size++;
			}
		}

		std::unique_lock<std::mutex> lock(v_lock);
		result[id] = tmp;
		lock.unlock();
		};

	for (int i = 0; i < iters; i++)
	{
		threads.push_back(std::thread(ransaciter, i));
	}

	for (int i = 0; i < iters; i++)
	{
		threads[i].join();
	}

	for (const parallelInlierStruct& p : result)
	{
		if (p.size > bestModel.size)
		{
			bestModel = p;
		}
	}

	return filterInliers(getCircleParams(filterInliers(bestModel.c, pts, threshold)), pts, threshold);
}

#pragma endregion

cudaError_t ransacCuda(const std::vector<float>& pts, std::vector<float>& circle_params, std::vector<int>& count);

__device__ void printMat(float** mat, int n, int m)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("%f ", mat[i][j]);
		}
		printf("\n");
	}
}

__device__ float distance(float* c, float* p)
{
	return fabsf(sqrtf(powf(p[0] - c[0], 2) + powf(p[1] - c[1], 2)) - c[2]);

}

__global__ void countInliers(float* pts, float** bs, int* count_inliers, float* circle_params, int size, float threshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	circle_params[id * 3] = bs[id][0] / 2.0f;
	circle_params[id * 3 + 1] = bs[id][1] / 2.0f;
	circle_params[id * 3 + 2] = sqrtf(bs[id][2] + powf(circle_params[id * 3], 2) + powf(circle_params[id * 3 + 1], 2));

	float c[3] = {circle_params[id * 3] , circle_params[id * 3 + 1] , circle_params[id * 3 + 2]};

	int count = 0;
	for (int i = 0; i < size; i++)
	{
		float p[2] = { pts[i * 2], pts[i * 2 + 1] };
		if (distance(c, p) < threshold)
			count++;
	}

	count_inliers[id] = count;
}

__global__ void createSystems(float* pts, float** As, float** bs, int seed, curandState* rand_state, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &rand_state[id]);

	int idx1, idx2, idx3;
	idx1 = curand_uniform(&(rand_state[id])) * size - 1;	//curand excludes 0.0 and includes 1.0
	idx2 = curand_uniform(&(rand_state[id])) * size - 1;
	idx3 = curand_uniform(&(rand_state[id])) * size - 1;

	//Put them in column major for cublas
	//x, y, 1
	As[id][0] = pts[idx1 * 2];
	As[id][1] = pts[idx2 * 2];
	As[id][2] = pts[idx3 * 2];
	As[id][3] = pts[idx1 * 2 + 1];
	As[id][4] = pts[idx2 * 2 + 1];
	As[id][5] = pts[idx3 * 2 + 1];
	As[id][6] = 1;
	As[id][7] = 1;
	As[id][8] = 1;

	//X^2 + y^2
	bs[id][0] = pts[idx1 * 2] * pts[idx1 * 2] + pts[idx1 * 2 + 1] * pts[idx1 * 2 + 1];
	bs[id][1] = pts[idx2 * 2] * pts[idx2 * 2] + pts[idx2 * 2 + 1] * pts[idx2 * 2 + 1];
	bs[id][2] = pts[idx3 * 2] * pts[idx3 * 2] + pts[idx3 * 2 + 1] * pts[idx3 * 2 + 1];

	//printMat(As, 3, 3);
	//printMat(bs, 3, 3);
}

int main()
{
	//Init random
	udist = std::uniform_real_distribution<float>(0.f, 1.f);
	rng.seed(static_cast<long unsigned int>(time(0)));

#pragma region Get input

	std::vector<Data> input;
	int u, v;
	for (int i = 1; i <= img_count; i++)
	{
		std::vector<cv::Point2i> tmpPoints;

		std::ifstream f("Assets/points" + std::to_string(i) + ".txt");

		if (!f.is_open())
			return 1;

		//Fapados kommentkiszedés fájl elejéről
		f.ignore(1000, '\n');

		while (!f.eof())
		{
			f >> u >> v;
			tmpPoints.push_back({ u, v });
		}

		f.close();
		input.push_back({ cv::imread("Assets/image" + std::to_string(i) + ".png"), tmpPoints });
	}

#pragma endregion
	
	for (int i = 0; i < input.size(); i++)
	{
		Data d = input[i];

		std::vector<cv::Point2f> norm;
		for (cv::Point2i p : d.points)
		{
			norm.push_back(p);
		}

		cv::Mat T = normalize(norm);

		std::vector<float> input(norm.size() * 2);
		std::vector<float> circle_output(iters * 3);
		std::vector<int> count_output(iters);
		for (int i = 0; i < norm.size(); i++)
		{
			input[i * 2] = norm[i].x;
			input[i * 2 + 1] = norm[i].y;
		}

		cudaError_t cudaStatus = ransacCuda(input, circle_output, count_output);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "ransacCuda failed!");
			return 1;
		}

		int mostInlierIndex = 0;
		for (int i = 0; i < iters; i++)
		{
			if (count_output[i] > count_output[mostInlierIndex])
				mostInlierIndex = i;
		}

		cv::Point3f bestModel = { circle_output[mostInlierIndex * 3], circle_output[mostInlierIndex * 3 + 1], circle_output[mostInlierIndex * 3 + 2] };

		std::cout << "Most inliers: " << count_output[mostInlierIndex] << "\n";
		std::cout << "Model: " << bestModel.x << " " << bestModel.y << " " << bestModel.z << "\n";

		try
		{
			std::vector<cv::Point2f> gpuPoints = filterInliers(getCircleParams(filterInliers(bestModel, norm, threshold)), norm, threshold);
			std::vector<cv::Point3f> phsGpu;
			cv::convertPointsToHomogeneous(gpuPoints, phsGpu);
			for (cv::Point3f p : phsGpu)
			{
				cv::Mat tmp(p);
				tmp = T.inv() * tmp;
				tmp = tmp * (1.0 / tmp.at<float>(2, 0));
				cv::Point2f imgPoint = { tmp.at<float>(0, 0), tmp.at<float>(1, 0) };

				cv::circle(d.img, imgPoint, 4, { 255, 0, 0 }, 2);
			}

			cv::imwrite("Assets/gpu_circle_" + std::to_string(i + 1) + ".jpg", d.img);
		}
		catch (cv::Exception e)
		{
			std::cout << "CV exception in final model fitting" << std::endl;
		}

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    return 0;
}

cudaError_t ransacCuda(const std::vector<float>& pts, std::vector<float>& circle_params, std::vector<int>& count)
{
	//Point coordinates already separated into two float data
	float* d_pts = 0;
	float* d_circle_params = 0;
	int* h_count = (int*)malloc(sizeof(int) * iters);
	int* d_count = 0;
	float** d_As;
	float** d_bs;
	float** h_As = (float**)malloc(iters * sizeof(float*));
	float** h_bs = (float**)malloc(iters * sizeof(float*));
	curandState* d_state;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers
	cudaMalloc(&d_state, iters * sizeof(curandState));
	cudaMalloc((void**)&d_pts, pts.size() * sizeof(float));
	cudaMalloc((void**)&d_circle_params, iters * 3 * sizeof(float));
	cudaMalloc((void**)&d_count, iters * sizeof(int));
	cudaMalloc((void**)&d_As, iters * sizeof(float*));
	cudaMalloc((void**)&h_As[0], iters * 9 * sizeof(float));
	cudaMalloc((void**)&d_bs, iters * sizeof(float*));
	cudaMalloc((void**)&h_bs[0], iters * 3 * sizeof(float));

	for (int i = 1; i < iters; i++)
	{
		h_As[i] = h_As[i - 1] + 9;
		h_bs[i] = h_bs[i - 1] + 3;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(d_As, h_As, iters * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bs, h_bs, iters * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

	srand(time(0));
	int seed = rand();
	createSystems <<<iters / threadNum, threadNum>>> (d_pts, d_As, d_bs, seed, d_state, pts.size());

#pragma region Least squares fuckery
	cublasHandle_t cublasH;

	int* d_info = nullptr;

	cublasCreate(&cublasH);

	cublasSgelsBatched(cublasH, cublasOperation_t::CUBLAS_OP_N, 3, 3, 1, d_As, 3, d_bs, 3, d_info, NULL, iters);
#pragma endregion

	countInliers <<<iters / threadNum, threadNum>>> (d_pts, d_bs, d_count, d_circle_params, pts.size(), threshold);
	cudaDeviceSynchronize();
	cudaMemcpy(h_count, d_count, iters * sizeof(int), cudaMemcpyDeviceToHost);
	float* h_circle_params = (float*)malloc(sizeof(float) * 3 * iters);
	cudaMemcpy(h_circle_params, d_circle_params, 3 * sizeof(float) * iters, cudaMemcpyDeviceToHost);

	for (int i = 0; i < iters; i++)
	{
		count[i] = h_count[i];
		circle_params[i * 3] = h_circle_params[i * 3];
		circle_params[i * 3 + 1] = h_circle_params[i * 3 + 1];
		circle_params[i * 3 + 2] = h_circle_params[i * 3 + 2];
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ransacIterCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching ransacIterCuda!\n", cudaStatus);
		goto Error;
	}

Error:
	cublasDestroy(cublasH);

	cudaFree(d_As);
	cudaFree(d_bs);
	cudaFree(d_pts);
	cudaFree(d_circle_params);
	cudaFree(d_count);
	cudaFree(d_state);

	free(h_bs);
	free(h_As);
	free(h_circle_params);
	free(h_count);

	return cudaStatus;
}
