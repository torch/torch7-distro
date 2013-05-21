FIND_PACKAGE(Torch REQUIRED)
SET(src testwrap.c THTestWrap.h THTestWrap.c)
FILE(GLOB luasrc testwrap.lua)

ADD_TORCH_WRAP(testwrap_wrap testwrap.lua)

ADD_TORCH_PACKAGE(testwrap "${src}" "${luasrc}" "Tests for wrap.lua")
TARGET_LINK_LIBRARIES(testwrap luaT TH)
