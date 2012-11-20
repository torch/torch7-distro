# Check if the processor is an ARM and if Neon instruction are available on the machine where
# the project is compiled.

IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
   EXEC_PROGRAM(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)

   #neon instruction can be found on the majority part of modern ARM processor
   STRING(REGEX REPLACE "^.*(neon).*$" "\\1" NEON_THERE ${CPUINFO})
   STRING(COMPARE EQUAL "neon" "${NEON_THERE}" NEON_TRUE)
   IF (NEON_TRUE)
      set(NEON_FOUND true CACHE BOOL "NEON available on host")
   ELSE (NEON_TRUE)
      set(NEON_FOUND false CACHE BOOL "NEON available on host")
   ENDIF (NEON_TRUE)

ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
   EXEC_PROGRAM("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE
      CPUINFO)

   #neon instruction can be found on the majority part of modern ARM processor
   STRING(REGEX REPLACE "^.*(neon).*$" "\\1" NEON_THERE ${CPUINFO})
   STRING(COMPARE EQUAL "neon" "${NEON_THERE}" NEON_TRUE)
   IF (NEON_TRUE)
      set(NEON_FOUND true CACHE BOOL "NEON available on host")
   ELSE (NEON_TRUE)
      set(NEON_FOUND false CACHE BOOL "NEON available on host")
   ENDIF (NEON_TRUE)

ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
   # TODO
   set(NEON_FOUND   false CACHE BOOL "NEON not available on host")
ELSE(CMAKE_SYSTEM_NAME MATCHES "Linux")
   set(NEON_FOUND   false CACHE BOOL "NEON not available on host")
ENDIF(CMAKE_SYSTEM_NAME MATCHES "Linux")

if(NEON_FOUND)
      MESSAGE(STATUS "Find hardware support for NEON on this machine.")
endif(NEON_FOUND)
mark_as_advanced(NEON_FOUND)
