const validateBase = require('./validateBase');

const subject = { name: 'John', age: 30 };
const schema = { type: 'object', properties: { name: { type: 'string' }, age: { type: 'number' } } };

const valid = validateBase({ schema, subject }, subject, schema);

if (valid) {
  console.log('Subject is valid.');
} else {
  console.log('Subject is invalid.');
}
