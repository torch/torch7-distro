local interface = wrap.CInterface.new()

interface:print(
   [[
#include "luaT.h"
#include "TH.h"
#include "THTestWrap.h"
   ]])

--[[
interface:wrap('wraptwodouble',
               'THTestWrap_twodouble',
               {{name="long"},
               {name="long"}
           })

interface:wrap('wraponedouble',
               'THTestWrap_onedouble',
               {{name="long"}}
           )
--]]
interface:wrap('wraponedoublecreturned',
               'THTestWrap_onedoublecreturned',
               {{name="long", creturned=true}}
           )

interface:register("testwrap__")

interface:print(
   [[
int luaopen_libtestwrap(lua_State *L)
{

  luaL_register(L, "libtestwrap", testwrap__);

  return 1;
}
   ]])

interface:tofile(arg[1])
