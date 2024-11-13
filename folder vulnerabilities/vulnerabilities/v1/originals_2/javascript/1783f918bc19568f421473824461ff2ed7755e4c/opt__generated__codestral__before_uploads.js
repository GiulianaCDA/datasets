'use strict';

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

    if (!socket.uid || !data || !data.chunk || !data.params) {
        throw new Error('[[error:invalid-data]]');
    }

    const method = data.params.method;

    if (typeof method !== 'string' || !(method in methodToFunc)) {
        throw new Error('[[error:invalid-method]]');
    }

    // ... rest of the code
};

uploads.clear = function (sid) {
    delete inProgress[sid];
};
