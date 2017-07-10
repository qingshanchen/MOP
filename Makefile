CFLAGS	         = -fPIC  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fvisibility=hidden -O3
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =

CCINCLUDES = -I/home/qsc/mylib/petsc-3.7.6/include -I/home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt/include -I/usr/include
#LIBS = -Wl,-rpath,/home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt/lib -L/home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt/lib -Wl,-rpath,/home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/6 -L/usr/lib/gcc/x86_64-linux-gnu/6 -Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -lpetsc -llapack -lblas -lexoIIv2for -lexodus -lnetcdf -lhdf5hl_fortran -lhdf5_fortran -lhdf5_hl -lhdf5 -ltriangle -lX11 -lpthread -lm -lmpichfort -lgfortran -lm -lgfortran -lm -lquadmath -lmpichcxx -lstdc++ -lm -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/6 -L/usr/lib/gcc/x86_64-linux-gnu/6 -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lmpich -lgcc_s -ldl  -I/usr/include -L/usr/lib -lnetcdf
LIBS = -Wl,-rpath,/home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt/lib -L/home/qsc/mylib/petsc-3.7.6/arch-linux2-c-opt/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/6 -L/usr/lib/gcc/x86_64-linux-gnu/6 -lpetsc -llapack -lblas -lexoIIv2for -lexodus -lnetcdf -lhdf5hl_fortran -lhdf5_fortran -lhdf5_hl -lhdf5 -ltriangle -lX11 -lpthread -lm -lmpichfort -lgfortran -lm -lgfortran -lquadmath -lmpichcxx -lstdc++ -lm -lmpich -lgcc_s -ldl  -I/usr/include -L/usr/lib -lnetcdf

default: mop

mop: mop.o 
	mpicc -o mop mop.o  ${LIBS}

mop.o: mop.c
	mpicc ${CFLAGS} ${CCINCLUDES} -c -o mop.o mop.c

clean::
	${RM} mop
	${RM} mop.o
