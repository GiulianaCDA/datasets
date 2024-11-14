```php
<?php
if (!defined('FLUX_ROOT')) exit;
$this->loginRequired(); 
?>

<?php if($ticketlist): ?>
<h2><?php echo Flux::message('SDHeaderID') ?><?php echo htmlspecialchars($trow->ticket_id) ?> - <?php echo htmlspecialchars($trow->subject) ?></h2>
	<table class="vertical-table" width="100%"> 
		<tbody>
		<tr>
			<th>Account</th>
				<td><?php echo $this->linkToAccount($trow->account_id,$session->account->userid . ' (' . $session->account->account_id . ')') ?></td>
			<th>Characters Affected</th>
				<?php if($trow->char_id === '0'):?>
				<td><i>All Characters</i></td>
				<?php elseif($trow->char_id === '-1'):?>
				<td><i>None on account</i></td>
				<?php else:?>
				<td><?php echo $this->linkToCharacter($trow->char_id,$char->name) ?></td>
				<?php endif ?>
		</tr>
		<tr>
			<th>Status</th>
			<td><?php echo htmlspecialchars($trow->status) ?></td>
		</tr>
		<tr>
			<th>Action</th>
			<td><?php echo htmlspecialchars($trow->action) ?></td>
		</tr>
		<tr>
			<th>Timestamp</th>
			<td><?php echo htmlspecialchars($trow->timestamp) ?></td>
		</tr>
		</tbody>
	</table><br />
<?php else: ?>
	<p>
		<?php echo htmlspecialchars(Flux::message('SDHuh')) ?>
	</p>
<?php endif ?>
