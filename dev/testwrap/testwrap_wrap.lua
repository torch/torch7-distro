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

--]]
interface:wrap('ReturnOneDouble',
               'THTestWrap_ReturnOneDouble',
               {{name="double", returned=true, invisible=true, default=0}}
           )

interface:wrap('CReturnOneDouble',
               'THTestWrap_CReturnOneDouble',
               {{name="double", creturned=true}}
           )

interface:wrap('ReturnOneIndex',
               'THTestWrap_ReturnOneIndex',
               {{name="index", returned=true, invisible=true, default=0}}
           )

interface:wrap('ReturnOneBoolean',
               'THTestWrap_ReturnOneBoolean',
               {{name="boolean", returned=true, invisible=true, default=0}}
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
