PETSC_DIR = /home/qsc/mylib/petsc-3.7.6
#PETSC_ARCH = /home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt
PETSC_ARCH = /home/qsc/mylib/petsc-3.7.6/arch-linux2-c-debug


MPICC = ${PETSC_ARCH}/bin/mpicc

CFLAGS	         = -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =


CCINCLUDES = -I${PETSC_DIR}/include -I${PETSC_ARCH}/include
LIBS = -Wl,-rpath,${PETSC_ARCH}/lib -L${PETSC_ARCH}/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/6 -L/usr/lib/gcc/x86_64-linux-gnu/6 -lpetsc -llapack -lblas  -lX11 -lpthread -lm -lquadmath -lmpichcxx -lstdc++ -lm -lmpich -lgcc_s -ldl -lnetcdf

default: mop

mop: mop.o 
	${MPICC} ${CFLAGS} -o mop mop.o  ${LIBS}

mop.o: mop.c
	${MPICC} ${CFLAGS} ${CCINCLUDES} -c -o mop.o mop.c

clean::
	${RM} mop
	${RM} mop.o
