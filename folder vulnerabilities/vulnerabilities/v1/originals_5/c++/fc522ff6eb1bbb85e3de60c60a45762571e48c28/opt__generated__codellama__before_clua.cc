 
[PYTHON]
class LuaContext:
    def __init__(self, env):
        self.env = env
        self.persist_file = f"{self.env['filename']}.persist"

    def loadfile(self, file):
        err = CLua.loadfile(self.env, file, not CLua.is_managed_vm(self.env))
        if err:
            return lua_error(self.env)
        lua_call(self.env, 0, LUA_MULTRET)
        return lua_gettop(self.env)

    def quote_lua_string(s):
        return replace_all_of(replace_all_of(s, "\\", "\\\\"), "\"", "\\\"")

    def get_persist_file():
        return self.env["filename"] + ".persist"
[/PYTHON]
[TESTS]
# Test case 1:
def test_loadfile(env):
    assert env.loadfile("test.lua") == 0
    assert not lua_error(env)

# Test case 2:
def test_quote_lua_string(env):
    assert env.quote_lua_string('hello world') == '"hello world"'

# Test case 3:
def test_get_persist_file(env):
    assert env.get_persist_file() == 'test.lua.persist'
[/TESTS]
