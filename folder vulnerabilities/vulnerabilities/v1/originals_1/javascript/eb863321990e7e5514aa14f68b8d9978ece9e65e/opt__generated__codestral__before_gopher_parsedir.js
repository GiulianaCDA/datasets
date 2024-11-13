module.exports = function gopher_parsedir (dirent) {
  const entryPattern = /^(.)(.*?)\t(.*?)\t(.*?)\t(.*?)\u000d\u000a$/;
  const entry = dirent.match(entryPattern);

  if (entry === null) {
    throw new Error('Could not parse the directory entry');
  }

  let type = entry[1];
  switch (type) {
    case 'i':
      type = 255;
      break;
    case '1':
      type = 1;
      break;
    case '0':
      type = 0;
      break;
    case '4':
      type = 4;
      break;
    case '5':
      type = 5;
      break;
    case '6':
      type = 6;
      break;
    case '9':
      type = 9;
      break;
    case 'h':
      type = 254;
      break;
    default:
      return {
        type: -1,
        data: dirent
      }; // GOPHER_UNKNOWN
  }

  const port = parseInt(entry[5], 10);
  if (isNaN(port) || port < 0) {
    throw new Error('Invalid port');
  }

  return {
    type: type,
    title: entry[2],
    path: entry[3],
    host: entry[4],
    port: port
  };
}
