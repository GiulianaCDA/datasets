<?php
header('Content-type: text/javascript');
?>

/*------------------------------------------------------------------------------
 * End redefinition of window.alert function
 *----------------------------------------------------------------------------*/

function fieldInfo(id){
    alert(info[id]);
}

function testftp(form){
    var params = {
        '-action': 'testftp',
        '-ftphost': encodeURIComponent(form.ftp_host.value),
        '-ftpuser': encodeURIComponent(form.ftp_username.value),
        '-ftppass': encodeURIComponent(form.ftp_password.value),
        '-ftppath': encodeURIComponent(form.ftp_path.value)
    };
    
    postDataReturnText('installer.php', params, function(result){
        var resultDiv = document.getElementById('ftp-test-results');
        eval('var results = '+result+';');
        resultDiv.innerHTML = results.message;
    });
}

function testdb(form){
    var params = {
        '-action': 'testdb',
        '-dbname': encodeURIComponent(form.database_name.options[form.database_name.selectedIndex].value),
        '-dbuser': encodeURIComponent(form.mysql_user.value),
        '-dbpass': encodeURIComponent(form.mysql_password.value)
    };
    
    postDataReturnText('installer.php', params, function(result){
        var resultDiv = document.getElementById('db-test-results');
        eval('var results = '+result+';');
        resultDiv.innerHTML = results.message;
    });
}

var listeners = {
    'database_name': {
        'onchange': function(fld){
            var option = fld.options[fld.selectedIndex].value;
            if (option) {
                document.getElementById('step2').style.display = '';
            } else {
                document.getElementById('step2').style.display = 'none';
            }
        }
    },
    'install_type': {
        'onchange': function(fld){
            var option = fld.options[fld.selectedIndex].value;
            if (option == 'ftp_install') {
                document.getElementById('step4').style.display = '';
            } else {
                document.getElementById('step4').style.display = 'none';
                document.getElementById('submitstep').style.display = '';
            }
        }
    }
};
