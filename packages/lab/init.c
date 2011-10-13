#include "luaT.h"
#include "lab.h"

extern void lab_utils_init(lua_State *L);

DLL_EXPORT int luaopen_liblab(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "lab");

  lab_Byteinit(L);
  lab_Charinit(L);
  lab_Shortinit(L);
  lab_Intinit(L);
  lab_Longinit(L);
  lab_Floatinit(L);
  lab_Doubleinit(L);

  lab_init(L);
  lab_utils_init(L);

  return 1;
}
