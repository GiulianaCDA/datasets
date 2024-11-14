 <!DOCTYPE html>
<html lang="en">
<head>
    <?php $mpar = $params; ?>
    <?php if (isset($mpar['module_settings'])) {
        unset($mpar['module_settings']);
    }
    ?>

    <script type="text/javascript">
        mw.lib.require('bootstrap');
        mw.lib.require('mwui_dialog');
        mw.lib.require('mwui_forms');
        <?php if (isset($params['id'])): ?>
        var moduleId = '<?php print $params['id'] ?>';
        <?php endif; ?>
    </script>
    <style type="text/css">
        .mw-dialog-header {
            display: none !important;
        }

        #settings-container {
            padding: 15px 0 0 20px;
        }
    </style>
    <script type="text/javascript">
        mw.require('prop_editor.js');
        mw.require('forms.js');
        mw.require('options.js');
        mw.require('module_settings.js');
        <?php if (isset($params['id'])): ?>
        mw.require('tools.js', true);
        mw.lib.require('events');
        $(document).ready(function () {
            var moduleId = '<?php print $params['id'] ?>';
            var parent_window = window.parent;

            if (typeof(moduleId) != "undefined") {
                mw.on('mw:load', function () {
                    mw.$("#" + moduleId).liveEdit();
                    mw.$("#" + moduleId).data('live-edit').bind('settingsChanged', settingsAction);
                    if (parent_window != window) {
                        parent_window.mw.on('mw:load', function () {
                            var parentModule = parent_window.mw.$("#" + moduleId).data('live-edit');
                            if (parentModule) {
                                mw.$("#" + moduleId).data('live-edit').bind('settingsChanged', settingsAction);
                            }
                        });
                    }
                })
            }

            function loadEditModule(id, params, value) {
                var parent_window = window.parent;
                if (typeof(parent_window.$) != 'undefined') {
                    mw.$("#" + id).data('live-edit').loadModule(params, value);
                } else {
                    setTimeout(function () {
                        loadEditModule(id, params, value);
                    }, 500)
                }
            }

            function modal_preset_manager_html_placeholder_for_reload() {
                var placeholder = window.parent.document.querySelector('#module-id-<?php print $params['id'] ?> .mw-module-settings-liveedit-holder');
                if (placeholder && typeof(window.parent.$) != 'undefined') {
                    var module_html = $(placeholder).html();
                    window.parent.$('.mw-module-live-edit-settings').each(function () {
                        if ($(this).attr('id') == 'module-id-<?php print $params['id'] ?>') {
                            $(this).find('.mw-module-settings-liveedit-holder').html(module_html);
                        }
                    });
                } else if (placeholder) {
                    setTimeout(function () {
                        modal_preset_manager_html_placeholder_for_reload()
                    }, 1000)
                }
            }

            function module_preset_linked_dd_menu_show_icon() {
                var parent_window = window.parent;
                if (typeof(parent_window.$) != 'undefined') {
                    parent_window.$('.mw-module-settings-liveedit-holder').each(function () {
                        $(this).find('.mw-presets-dropdown-menu .dd-item:first-child').addClass('active');
                        if ($(this).find('.module-preset-linked-dd-menu .dd-item.active').length > 0) {
                            var parent_holder = $(this).parents('.mw-module-settings-liveedit-holder:first');
                            parent_holder.find('.mw-presets-dropdown .dd-handle i.fa').removeClass('fa-plus-circle').addClass('fa-pencil');
                        }
                    });
                } else {
                    setTimeout(function () {
                        module_preset_linked_dd_menu_show_icon()
                    }, 100)
                }
            }

            function modal_preset_manager_html_placeholder_for_reload() {
                var placeholder = window.parent.document.querySelector('#module-id-<?php print $params['id'] ?> .mw-module-settings-liveedit-holder');
                if (placeholder && typeof(window.parent.$) != 'undefined') {
                    var module_html = $(placeholder).html();
                    window.parent.$('.mw-module-live-edit-settings').each(function () {
                        if ($(this).attr('id') == 'module-id-<?php print $params['id'] ?>') {
                            $(this).find('.mw-module-settings-liveedit-holder').html(module_html);
                        }
                    });
                } else if (placeholder) {
                    setTimeout(function () {
                        modal_preset_manager_html_placeholder_for_reload()
                    }, 1000)
                }
            }

            function module_preset_linked_dd_menu_show_icon() {
                var parent_window = window.parent;
                if (typeof(parent_window.$) != 'undefined') {
                    parent_window.$('.mw-module-settings-liveedit-holder').each(function () {
                        $(this).find('.mw-presets-dropdown-menu .dd-item:first-child').addClass('active');
                        if ($(this).find('.module-preset-linked-dd-menu .dd-item.active').length > 0) {
                            var parent_holder = $(this).parents('.mw-module-settings-liveedit-holder:first');
                            parent_holder.find('.mw-presets-dropdown .dd-handle i.fa').removeClass('fa-plus-circle').addClass('fa-pencil');
                        }
                    });
                } else {
                    setTimeout(function () {
                        module_preset_linked_dd_menu_show_icon()
                    }, 100)
                }
            }

            $(document).on('click', '.mw-presets-dropdown .dd-item:not(".create-preset")', function () {
                var parent_window = window.parent;
                if (typeof(parent_window.$) != 'undefined') {
                    parent_window.$('.mw-module-live-edit-settings').each(function () {
                        if ($(this).attr('id') == 'module-id-<?php print $params['id'] ?>') {
                            var active_preset = $(this).find('.mw-presets-dropdown-menu .dd-item.active');
                            var active_preset_data = active_preset.attr('data-settings');
                            if (typeof(active_preset_data) != 'undefined') {
                                $(this).find('.mw-presets-dropdown').data('current', active_preset_data);
                            }
                        }
                    });
                }
            });

            <?php if (isset($params['module'])): ?>
            mw.on('mw:load', function () {
                var dd = $(".mw-presets-dropdown");
                if (!dd.data('settings-loaded')) {
                    dd.data('settings-loaded', true);
                    var parent_window = window.parent;
                    if (typeof(parent_window.$) != 'undefined') {
                        var currentPreset = parent_window.$("#" + moduleId).data("current");
                        if (currentPreset && dd.find('[data-settings="' + currentPreset + '"]').length > 0) {
                            dd.find('.dd-handle .title').html(dd.find('[data-settings="' + currentPreset + '"]').html());
                        }
                    } else if (dd.find('[data-settings]').length == 1) {
                        var data = dd.find('[data-settings]').attr("data-settings");
                        dd.find('.dd-handle .title').html(dd.find('[data-settings="' + currentPreset + '"]').html());
                        dd.data('current', data);
                    }
                }
            });

            var moduleSettingsLoaded = false;
            $(document).on("click", "#mw_module_manager_edit_<?php print $params['id']; ?>", function () {
                if (!moduleSettingsLoaded) {
                    setTimeout(function () {
                        var dd = $(".mw-presets-dropdown");
                        if (dd.length > 0) {
                            moduleSettingsLoaded = true;
                            var parent_window = window.parent;
                            if (typeof(parent_window.$) != 'undefined') {
                                var currentPreset = parent_window.$("#" + moduleId).data("current");
                                if (currentPreset && dd.find('[data-settings="' + currentPreset + '"]').length > 0) {
                                    dd.find('.dd-handle .title').html(dd.find('[data-settings="' + currentPreset + '"]').html());
                                }
                            } else if (dd.find('[data-settings]').length == 1) {
                                var data = dd.find('[data-settings]').attr("data-settings");
                                dd.find('.dd-handle .title').html(dd.find('[data-settings="' + currentPreset + '"]').html());
                                dd.data('current', data);
                            }
                        }
                    }, 100);
                }
            });

            <?php endif; ?>
            $(document).on("click", "#mw_module_manager_edit_<?php print $params['id']; ?>", function () {
                modal_preset_manager_html_placeholder_for_reload();
                module_preset_linked_dd_menu_show_icon();
            });

            <?php if (isset($params['module'])): ?>
            $(document).on("click", "#mw_module_manager_delete_<?php print $params['id']; ?>", function () {
                var parent_window = window.parent;
                if (typeof(parent_window.$) != 'undefined') {
                    parent_window.$('.mw-module-live-edit-settings').each(function () {
                        if ($(this).attr('id') == 'module-id-<?php print $params['id'] ?>') {
                            $(this).find('.mw-presets-dropdown .dd-handle i.fa').removeClass('fa-pencil').addClass('fa-plus-circle');
                        }
                    });
                } else {
                    setTimeout(function () {
                        module_preset_linked_dd_menu_show_icon()
                    }, 100)
                }
            });
            <?php endif; ?>

            $(document).on('click', '.mw-presets-dropdown .create-preset', function () {
                var parent_window = window.parent;
                if (typeof(parent_window.$) != 'undefined') {
                    parent_window.$('.mw-module-live-edit-settings').each(function () {
                        if ($(this).attr('id') == 'module-id-<?php print $params['id'] ?>') {
                            $(this).find('.mw-presets-dropdown .dd-handle i.fa').removeClass('fa-plus-circle').addClass('fa-pencil');
                        }
                    });
                } else {
                    setTimeout(function () {
                        module_preset_linked_dd_menu_show_icon()
                    }, 100)
                }
            });
        })
        <?php endif; ?>

    </script>