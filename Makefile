OPTS = -Wall -g -O3 -Wno-deprecated
LIB = -lboost_system

all: shooterDAggerUnrolled shooterDAggerUndiscounted shooterDAggerMCTSUnrolled shooterDAggerMCTSUndiscounted

shooterDAggerMCTSUndiscounted: shooterDAggerMCTSUndiscounted.cc ShooterModel.o SamplingModel.h ConvolutionalBinaryCTS.o PatchRewardModel.o RewardModel.h ShooterRewardModel.o cts.o PowFast.o icsilog.o icsilogw.hpp jacoblog.hpp
	g++ ${OPTS} -o shooterDAggerMCTSUndiscounted shooterDAggerMCTSUndiscounted.cc ShooterModel.o ConvolutionalBinaryCTS.o cts.o PatchRewardModel.o ShooterRewardModel.o PowFast.o icsilog.o ${LIB}

shooterDAggerMCTSUnrolled: shooterDAggerMCTSUnrolled.cc ShooterModel.o SamplingModel.h ConvolutionalBinaryCTS.o cts.o PowFast.o icsilog.o icsilogw.hpp jacoblog.hpp
	g++ ${OPTS} -o shooterDAggerMCTSUnrolled shooterDAggerMCTSUnrolled.cc ShooterModel.o ConvolutionalBinaryCTS.o cts.o PowFast.o icsilog.o ${LIB}

shooterDAggerUndiscounted: shooterDAggerUndiscounted.cc ShooterModel.o SamplingModel.h ConvolutionalBinaryCTS.o cts.o PowFast.o icsilog.o icsilogw.hpp jacoblog.hpp
	g++ ${OPTS} -o shooterDAggerUndiscounted shooterDAggerUndiscounted.cc ShooterModel.o ConvolutionalBinaryCTS.o cts.o PowFast.o icsilog.o ${LIB}

shooterDAggerUnrolled: shooterDAggerUnrolled.cc ShooterModel.o SamplingModel.h ConvolutionalBinaryCTS.o cts.o PowFast.o icsilog.o icsilogw.hpp jacoblog.hpp
	g++ ${OPTS} -o shooterDAggerUnrolled shooterDAggerUnrolled.cc ShooterModel.o ConvolutionalBinaryCTS.o cts.o PowFast.o icsilog.o ${LIB}

ShooterRewardModel.o: ShooterRewardModel.cc ShooterRewardModel.h
	g++ ${OPTS} -c ShooterRewardModel.cc

PatchRewardModel.o: PatchRewardModel.cc PatchRewardModel.h
	g++ ${OPTS} -c PatchRewardModel.cc

ConvolutionalBinaryCTS.o: ConvolutionalBinaryCTS.cc ConvolutionalBinaryCTS.h SamplingModel.h cts.hpp common.hpp
	g++ ${OPTS} -c ConvolutionalBinaryCTS.cc

ShooterModel.o: ShooterModel.cc ShooterModel.h SamplingModel.h
	g++ ${OPTS} -c ShooterModel.cc

cts.o: cts.cpp cts.hpp common.hpp PowFast.hpp icsilog.h icsilogw.hpp jacoblog.hpp
	g++ ${OPTS} -c cts.cpp

fastmath.o: fastmath.cpp fastmath.hpp jacoblog.hpp icsilogw.hpp PowFast.hpp
	g++ ${OPTS} -c fastmath.cpp

PowFast.o: PowFast.cpp PowFast.hpp
	g++ ${OPTS} -c PowFast.cpp

icsilog.o: icsilog.cpp icsilog.h
	g++ ${OPTS} -c icsilog.cpp

clean:
	rm *.o shooterDAggerUndiscounted shooterDAggerUnrolled shooterDAggerMCTSUndiscounted shooterDAggerMCTSUnrolled

cleanmac:
	rm -r *.dSYM
