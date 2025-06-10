To compile the program:
	g++ *.cpp -o HyperCircles -O3 -ffast-math -fopenmp -funroll-loops -march=native
	OR
	cmake cmake-build-debug	

Using the program tips:
	- you can save a generated set of HC's. But to use it again later for classification, you must import the training dataset.
	- If you want to change the distance metric we are using, go into Utils.h and simply change the #define NORM to 1 2 or 3 for whichever you like. Then recompile.
	- You must recompile after changing the distance metric, and saved HC's must be used with the distance metric they were generated with, or else you will get crazy results.
	- To change the classification voting system, go into testAccuracy, and change the argument to classifyPoint for HyperCircle::<voting_method> in the classificationMode argument.
		* the submode argument to that function is used in the USE_CIRCLES case of the classification mode. You use the submode to determine which voting style.
		* the classificationMode can be changed from USE_CIRCLES in the case that you want to classify points which the circles didn't cover.

Program should run on anything which has a g++ compiler. Though it was not tested WITHOUT openMP. There may be a special case where if OpenMP is not available or linked, the algorithms may not function properly.
