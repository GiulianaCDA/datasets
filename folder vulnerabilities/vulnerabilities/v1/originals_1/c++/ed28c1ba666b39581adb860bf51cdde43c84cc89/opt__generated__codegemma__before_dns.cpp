// Create a new resolver
Resolver* resolver = new Resolver(module);

// Add the resolver to the DNS class
DNS::AddResolverClass(resolver);
