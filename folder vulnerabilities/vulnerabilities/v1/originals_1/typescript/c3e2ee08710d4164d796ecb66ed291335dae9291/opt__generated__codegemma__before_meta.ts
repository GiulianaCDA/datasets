// Add metadata to a class using the @meta decorator
@meta({ key: 'author', value: 'John Doe' })
class MyClass {}

// Get the metadata record for the MyClass object
const metadata = MyClass.metadata;

// Print the author value
console.log(metadata.author); // Output: John Doe
