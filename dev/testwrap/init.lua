require 'libtestwrap'
testwrap = {}

local tester = torch.Tester()

function testwrap.testCReturn()
    local a = libtestwrap.CReturnOneDouble()
    tester:asserteq(a, 19, 'Wrong number returned')
end

tester:add(testwrap)
tester:run()

-- Add the tester to package to allow reruns
testwrap.tester = tester;

return testwrap
