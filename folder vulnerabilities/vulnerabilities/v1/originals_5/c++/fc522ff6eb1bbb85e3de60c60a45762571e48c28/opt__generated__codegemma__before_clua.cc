// Create a CLua instance.
CLua lua;

// Load a Lua script.
lua.dofile("script.lua");

// Create a Lua data structure.
lua_datum table(lua, -1);

// Access a value in the table.
lua_datum value(table, "key");

// Print the value.
std::cout << value.to_string() << std::endl;
