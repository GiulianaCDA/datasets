<script>
var grid = $("#ipv4-search").bootgrid({
    ajax: true,
    rowCount: [50, 100, 250, -1],
    templates: {
        header: "<div id=\"{{ctx.id}}\" class=\"{{css.header}}\"><div class=\"row\">"+
                "<div class=\"col-sm-9 actionBar\"><span class=\"pull-left\">"+
                "<form method=\"post\" action=\"\" class=\"form-inline\" role=\"form\">"+
                "<input type='hidden' name='csrf_token' value='<?php echo htmlspecialchars(csrf_token()); ?>'> "+
                // Rest of the form...
    },
    post: function () {
        return {
            id: "address-search",
            search_type: "ipv4",
            device_id: $('#device_id').val(),
            interface: $('#interface').val(),
            address: $('#address').val()
        };
    },
    // Rest of the script...
});
</script>
