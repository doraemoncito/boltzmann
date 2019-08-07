#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

double wtime() {
#ifdef _WIN32
	return timeGetTime() / 1000;
#else
#if defined(_OPENMP) && (_OPENMP > 200010)
	/* Use omp_get_wtime() if we can */
	return omp_get_wtime();
#else
	/* Use a generic timer */
	static int sec = -1;
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	if (sec < 0) sec = tv.tv_sec;
	return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
#endif
#endif
}
