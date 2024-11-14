// Create a bencoded dictionary
lazy_entry dictionary;
dictionary.dict_set("key1", "value1");
dictionary.dict_set("key2", 123);

// Print the dictionary to a string
std::string output = print_entry(dictionary, false, 0);

// Output: {"key1": "value1", "key2": 123}
