local interface = wrap.CInterface.new()

interface:print(
   [[
#include "luaT.h"
#include "TH.h"
   ]])

for _,name in ipairs({"seed", "initialSeed"}) do
   interface:wrap(name,
                  string.format("THRandom_%s",name),
                  {{name="long", creturned=true}})
end


interface:wrap('manualSeed',
               'THRandom_manualSeed',
               {{name="long"}})

interface:wrap('_setRNGState',
               'THRandom_setState',
               {{name="LongTensor"},
                {name="long"},
                {name="long"}})

interface:print(
    [[
static int wrapper_getRNGState(lua_State *L)
{
int narg = lua_gettop(L);
if(narg > 0)
    luaL_error(L, "expected arguments: none");

THLongTensor *arg1 = THLongTensor_newWithSize1d(624);
long arg2 = 0;
long arg3 = 0;

THRandom_getState(arg1,&arg2,&arg3);

luaT_pushudata(L, arg1, "torch.LongTensor");
lua_pushnumber(L, (lua_Number)arg2);
lua_pushnumber(L, (lua_Number)arg3);
return 3;
}
    ]])

interface:register("random__")
                
interface:print(
   [[
void torch_random_init(lua_State *L)
{
  luaL_register(L, NULL, random__);
  static const struct luaL_reg randomExtra__ [] = {
        {"_getRNGState", wrapper_getRNGState}
    };
  luaL_register(L, NULL, randomExtra__);
}
   ]])

interface:tofile(arg[1])
