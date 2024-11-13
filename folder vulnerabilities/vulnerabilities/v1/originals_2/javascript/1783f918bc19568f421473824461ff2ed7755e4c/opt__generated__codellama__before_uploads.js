const socketUser = require('./user');
const socketGroup = require('./groups');
const image = require('../image');
const meta = require('../meta');

const inProgress = {};

const uploads = module.exports;

uploads.upload = async function (socket, data) {
	const methodToFunc = {
		'user.uploadCroppedPicture': socketUser.uploadCroppedPicture,
		'user.updateCover': socketUser.updateCover,
		'groups.cover.update': socketGroup.cover.update,
	};

	const dataParams = JSON.parse(data.params);

	if (!socket.uid || !dataParams || !dataParams.method || !methodToFunc[dataParams.method]) {
		throw new Error('[[error:invalid-data]]');
	}

	inProgress[socket.id] = inProgress[socket.id] || {};
	const socketUploads = inProgress[socket.id];
	const { method } = dataParams;

	socketUploads[method] = socketUploads[method] || { imageData: '' };
	socketUploads[method].imageData += data.chunk;

	try {
		const maxSize = dataParams.method === 'user.uploadCroppedPicture' ?
			meta.config.maximumProfileImageSize : meta.config.maximumCoverImageSize;
		const size = image.sizeFromBase64(socketUploads[method].imageData);

		if (size > maxSize * 1024) {
			throw new Error(`[[error:file-too-big, ${maxSize}]]`);
		}
		if (socketUploads[method].imageData.length < dataParams.size) {
			return;
		}
		dataParams.imageData = socketUploads[method].imageData;
		const result = await methodToFunc[dataParams.method](socket, dataParams);
		delete socketUploads[method];
		return result;
	} catch (err) {
		delete inProgress[socket.id];
		throw err;
	}
};

uploads.clear = function (sid) {
	delete inProgress[sid];
};
