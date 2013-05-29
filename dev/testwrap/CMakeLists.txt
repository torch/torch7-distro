FIND_PACKAGE(Torch REQUIRED)
SET(src testwrap_wrap.c THTestWrap.h THTestWrap.c)
FILE(GLOB luasrc init.lua testwrap_wrap.lua)

ADD_TORCH_WRAP(testwrap_wrap testwrap_wrap.lua)

ADD_TORCH_PACKAGE(testwrap "${src}" "${luasrc}" "Tests for wrap.lua")
TARGET_LINK_LIBRARIES(testwrap luaT TH)
