 def delete_tmp_dir(self):
        '''Removes tmp dir'''
        try:
            shutil.rmtree(self.config['tmp_dir'])
        except Exception as e:
            self.log_error('Could not remove %s' % (self.config['tmp_dir']))
            self._exit('Exiting', 3)

    def _get_archive_path(self):
        '''Generates path for the archive based on options supplied to
        sos-collector'''
        tmp = ''
        if not self.config['output']:
            if self.config['tmp_dir'] == '/var/tmp':
                timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
                archive_name = 'sos-collector-%s.tar.gz' % timestamp
            else:
                tmp = self.config['tmp_dir'] + '/'
                archive_name = 'sos-collector.tar.gz'
        else:
            archive_name = self.config['output']
            if re.search('/', archive_name):
                output_path = os.path.dirname(archive_name)
                if not os.access(output_path, os.W_OK):
                    msg = 'You do not have write access to %s' % (output_path)
                    self._exit(msg, 4)
            elif '/' not in archive_name:
                tmp = self.config['tmp_dir'] + '/'
        return os.path.join('/', tmp, archive_name)

    def validate_cluster(self):
        '''Validates the cluster to ensure that it can be managed by this
        version of sos-collector. This only works for clusters using a
        host file or a kerberized kubernetes cluster'''
        try:
            if self.config['cluster'].node_list():
                return True
        except Exception as e:
            msg = 'Could not retrieve node list from supplied configuration.'
            msg += '\nPlease ensure that you are using the correct '
            msg += 'configuration file for your cluster and try again'
            self.log_error(msg)
            self._exit(e, 5)
        return False

    def validate_tmpdir(self):
        '''Ensures that supplied tmp dir exists and is writable'''
        if not os.path.exists(self.config['tmp_dir']):
            msg = 'No such file or directory: %s' % (self.config['tmp_dir'])
            self.log_error(msg)
            sys.exit(6)
        elif not os.access(self.config['tmp_dir'], os.W_OK):
            msg = 'Permission denied to write to: %s' % (self.config['tmp_dir'])
            self.log_error(msg)
            sys.exit(7)