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

interface:wrap('getRandomState',
               'THRandom_getState',
               {{name="LongTensor", returned=true},
                {name="long", returned=true},
                {name="long", returned=true}})

interface:wrap('setRandomState',
               'THRandom_setState',
               {{name="LongTensor"},
                {name="long"},
                {name="long"}})

interface:register("random__")
                
interface:print(
   [[
void torch_random_init(lua_State *L)
{
  luaL_register(L, NULL, random__);
}
   ]])

interface:tofile(arg[1])
