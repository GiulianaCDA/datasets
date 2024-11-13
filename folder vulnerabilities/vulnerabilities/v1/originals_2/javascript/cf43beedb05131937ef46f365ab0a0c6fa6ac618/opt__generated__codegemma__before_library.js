const commentsModule = require('./comments');

commentsModule.init({
  router: app,
  middleware: middleware,
  controllers: controllers
}, () => {
  // Initialization complete
});
