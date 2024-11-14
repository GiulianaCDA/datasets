module.exports = function gopher_parsedir(dirent) {
  // eslint-disable-line camelcase
  const entryPattern = /^(.)(.*?)\t(.*?)\t(.*?)\t(.*?)\u000d\u000a$/;
  const entry = dirent.match(entryPattern);

  if (entry === null) {
    throw new Error('Could not parse the directory entry');
  }

  let type = entry[1];
  switch (type) {
    case 'i':
      // GOPHER_INFO
      type = 255;
      break;
    case '1':
      // GOPHER_DIRECTORY
      type = 1;
      break;
    case '0':
      // GOPHER_DOCUMENT
      type = 0;
      break;
    case '4':
      // GOPHER_BINHEX
      type = 4;
      break;
    case '5':
      // GOPHER_DOSBINARY
      type = 5;
      break;
    case '6':
      // GOPHER_UUENCODED
      type = 6;
      break;
    case '9':
      // GOPHER_BINARY
      type = 9;
      break;
    case 'h':
      // GOPHER_HTTP
      type = 254;
      break;
    default:
      return {
        type: -1,
        data: dirent
      }; // GOPHER_UNKNOWN
  }
  return {
    type: parseInt(type),
    title: entry[2],
    path: entry[3],
    host: entry[4],
    port: entry[5]
  };
};
