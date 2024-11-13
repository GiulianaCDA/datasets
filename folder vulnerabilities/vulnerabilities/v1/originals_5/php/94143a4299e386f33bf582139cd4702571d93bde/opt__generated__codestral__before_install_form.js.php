<?php
header('Content-Type: application/javascript'); // It's better to use 'application/javascript' for JavaScript content
?>

/*------------------------------------------------------------------------------
 * End redefinition of window.alert function
 *----------------------------------------------------------------------------*/

function fieldInfo(id){
    if (typeof info[id] !== 'undefined') { // Added a check to ensure that the id exists in the info object
        alert(info[id]);
    } else {
        console.error('Invalid ID:', id);
    }
}

function testftp(form){
    var data = '-action=testftp&-ftphost='+encodeURIComponent(form.ftp_host.value)+'&-ftpuser='+encodeURIComponent(form.ftp_username.value)+'&-ftppass='+encodeURIComponent(form.ftp_password.value)+'&-ftppath='+encodeURIComponent(form.ftp_path.value);
    postDataReturnText('installer.php', data, function(result){
        var resultDiv = document.getElementById('ftp-test-results');
        try {
            var results = JSON.parse(result); // Use JSON.parse instead of eval to avoid JavaScript code injection attacks
            resultDiv.innerHTML = results.message;
        } catch (e) {
            console.error('Invalid JSON:', result, e);
        }
    });
}

function testdb(form){
    var data = '-action=testdb&-dbname='+encodeURIComponent(form.database_name.options[form.database_name.selectedIndex].value)+'&-dbuser='+encodeURIComponent(form.mysql_user.value)+'&-dbpass='+encodeURIComponent(form.mysql_password.value);
    postDataReturnText('installer.php', data, function(result){
        var resultDiv = document.getElementById('db-test-results');
        try {
            var results = JSON.parse(result); // Use JSON.parse instead of eval to avoid JavaScript code injection attacks
            resultDiv.innerHTML = results.message;
        } catch (e) {
            console.error('Invalid JSON:', result, e);
        }
    });
}

var listeners = {
    'database_name' : {
        'onchange' : function(fld){
            var option = fld.options[fld.selectedIndex].value;
            document.getElementById('step2').style.display = option ? '' : 'none'; // Simplified the if-else statement
        }
    },
    'install_type': {
        'onchange': function(fld){
            var option = fld.options[fld.selectedIndex].value;
            document.getElementById('step4').style.display = option === 'ftp_install' ? '' : 'none'; // Simplified the if-else statement
            document.getElementById('submitstep').style.display = option !== 'ftp_install' ? '' : 'none'; // Show/hide submit step based on install type
        }
    }
};
