<div class="space-10"></div>
<div class="col-md-12 col-xs-12">

	<form method="post" name="filters" id="filters_form_open" class="form-control" action="<?php echo $t_action; ?>">

	<?php # CSRF protection added here - form submits data to other pages ?>
	<input type="hidden" name="token" value="<?php echo auth_get_token() ?>" />
	<input type="hidden" name="type" value="1" />
	<input type="hidden" name="view_type" value="<?php echo $f_view_type; ?>" />
	<?php
		if( $f_for_screen == false ) {
			print '<input type="hidden" name="print" value="1" />';
			print '<input type="hidden" name="offset" value="0" />';
		}
	?>

		<div class="widget-box widget-color-blue2">
			<div class="widget-header widget-header-small">
				<h4 class="widget-title lighter">
					<i class="ace-icon fa fa-filter"></i>
					<?php echo lang_get('filters') ?>
				</h4>

				<div class="widget-toolbar">
					<div class="widget-menu">
						<a href="#" data-action="settings" data-toggle="dropdown">
							<i class="ace-icon fa fa-bars bigger-125"></i>
						</a>
						<ul class="dropdown-menu dropdown-menu-right dropdown-yellow dropdown-caret dropdown-closer">
							<?php
								$t_url = 'view_filters_page.php?static=' . (int)$f_static;
								if( $t_named_filter ) {
									$t_url .= '&filter_id=' . $f_filter_id;
								}
								$t_url .= '&view_type=';
								filter_print_view_type_toggle( $t_url, $t_filter['_view_type'] );
							?>
						</ul>
					</div>
				</div>

			</div>
			<div class="widget-body">
				<div class="widget-main no-padding">

					<div class="table-responsive">
							<?php
							filter_form_draw_inputs( $t_filter, $f_for_screen, $f_static );
							?>
					</div>

					<div class="widget-toolbox padding-8 clearfix">
						<div class="btn-toolbar pull-left">
							<div class="form-inline">
								<label><?php echo lang_get( 'search' ) ?>&nbsp;
									<input type="text" id="filter-search-txt" class="input-sm" size="16"
										   name="<?php echo FILTER_PROPERTY_SEARCH ?>"
										   value="<?php echo string_attribute( $t_filter[FILTER_PROPERTY_SEARCH] ) ?>">
								</label>
							</div>
						</div>
						<div class="btn-toolbar pull-right">
							<div class="form-inline">
								<input type="submit" class="btn btn-primary btn-sm btn-white btn-round no-float" name="filter" value="<?php echo lang_get( 'use_query' )?>" />
							</div>
						</div>
					</div>

				</div>
			</div>
		</div>
	</form>
</div>
<?php
layout_page_end();
