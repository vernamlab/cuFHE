/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>

namespace cufhe {

inline
void CuSafeCall__(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		fprintf( stderr, "CuSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

inline
void CuCheckError__(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CuCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
	// More careful checking. However, this will affect performance.
	// Comment out if needed.
	//#define safer
	#ifdef SAFER
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "CuCheckError() with sync failed at %s:%i : %s\n", file,
				line, cudaGetErrorString(err));
		exit(-1);
	}
	#endif
	return;
}

// @brief Report error location and terminate, if "cudaError != SUCCESS".
#define CuSafeCall(err)	CuSafeCall__(err, __FILE__, __LINE__)

// @brief Report error location and terminate, if last "cudaError != SUCCESS".
#define CuCheckError()	CuCheckError__(__FILE__, __LINE__)

} // namespace cufhe
