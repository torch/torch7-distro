#ifndef LAB_INC
#define LAB_INC

#include "luaT.h"

void lab_Byteinit(lua_State *L);
void lab_Charinit(lua_State *L);
void lab_Shortinit(lua_State *L);
void lab_Intinit(lua_State *L);
void lab_Longinit(lua_State *L);
void lab_Floatinit(lua_State *L);
void lab_Doubleinit(lua_State *L);
void lab_init(lua_State *L);

#endif

