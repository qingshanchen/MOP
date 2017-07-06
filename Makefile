CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

mop: mop.o chkopts
	-${CLINKER} -o mop mop.o  ${PETSC_KSP_LIB}
	${RM} mop.o


clean::
	${RM} mop
