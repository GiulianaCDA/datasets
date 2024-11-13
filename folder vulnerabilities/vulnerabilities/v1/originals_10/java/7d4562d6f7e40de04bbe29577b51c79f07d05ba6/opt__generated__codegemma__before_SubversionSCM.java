// Get Subversion module locations
List<ModuleLocation> modules = SubversionSCM.getModuleLocations(remoteLocations, localLocations, depthOptions, isIgnoreExternals);

// Compare two Subversion authentication objects
boolean areAuthenticationsEqual = SubversionSCM.compareSVNAuthentications(auth1, auth2);

// Get revision from remote URL
SVNRevision revision = SubversionSCM.getRevisionFromRemoteUrl(remoteUrl);
