all:
	icc -x c++ -Wall -std=c++11 -fopenmp -O3 -ipo -no-prec-div -xAVX -o d2q9-bgk d2q9-bgk.c

serial:
	icc -x c++ -Wall -std=c++11 -O3 -ipo -no-prec-div -xAVX -o d2q9-bgk d2q9-bgk.c

clean:
	rm d2q9-bgk
