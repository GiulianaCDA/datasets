 <?php
// $Id$

/**
* Setup GUI class
*
* @author Sascha Hofmann <shofmann@databay.de>
*/
class ilSetupGUI
{
	public function init()
	{
		global $tpl;

		require_once 'Services/Setup/classes/class.ilClientIniFile.php';
		$this->setup = new ilClientIniFile(CLIENT_INI_PATH);

		include_once("./Services/Form/classes/class.ilPropertyFormGUI.php");
	}

	public function displayProxy()
	{
		global $tpl;
		$this->initProxyForm();

		if(isset($_POST['proxy_status']))
		{
			$this->form->setValuesByPost();
		}
		else
		{
			$settings = $this->setup->getClient()->getAllSettings();
			$this->form->setValues($settings);
		}

		$tpl->setVariable("SETUP_CONTENT", $this->form->getHTML());

		if($settings['proxy_status'] == true)
		{
			$this->setup->printProxyStatus($this->setup->client);
		}
	}

	/**
	 *
	 * Init proxy form.
	 *
	 * @access	public
	 *
	 */
	public function initProxyForm()
	{
		global $lng;

		$this->form = new ilPropertyFormGUI();
		$this->form->setTableWidth('100%');
		$this->form->setId("proxy");
		$this->form->addCommandButton("saveProxy", $lng->txt("save"));
		$this->form->addCommandButton("displayProxy", $lng->txt("cancel"));

		// Proxy availability switch
		$proxs = new ilCheckboxInputGUI($lng->txt('proxy_status'), 'proxy_status');
		$proxs->setInfo($lng->txt('proxy_status_info'));
		$proxs->setValue(1);
		$this->form->addItem($proxs);

		// Proxy availability
		$proxa = new ilCustomInputGUI('', 'proxy_availability');
		$proxs->addSubItem($proxa);

		// Proxy
		$prox = new ilTextInputGUI($lng->txt('proxy_host'), 'proxy_host');
		$prox->setInfo($lng->txt('proxy_host_info'));
		$proxs->addSubItem($prox);

		// Proxy Port
		$proxp = new ilTextInputGUI($lng->txt('proxy_port'), 'proxy_port');
		$proxp->setInfo($lng->txt('proxy_port_info'));
		$proxp->setSize(10);
		$proxp->setMaxLength(10);
		$proxs->addSubItem($proxp);

		// save and cancel commands
		$this->form->addCommandButton('saveProxy', $lng->txt('save'));
	}

	/**
	 *
	 * Save proxy settings
	 *
	 * @access	public
	 *
	 */
	public function saveProxy()
	{
		global $lng;

		$this->initProxyForm();
		$isFormValid = $this->form->checkInput();

		$new_settings['proxy_status'] = (int)$this->form->getInput('proxy_status');
		$new_settings['proxy_host'] = trim($this->form->getInput('proxy_host'));
		$new_settings['proxy_port'] = trim($this->form->getInput('proxy_port'));

		if($isFormValid)
		{
			if($new_settings['proxy_status'] == true)
			{
				if(!strlen($new_settings['proxy_host']))
				{
					$isFormValid = false;
					$this->form->getItemByPostVar('proxy_host')->setAlert($lng->txt("msg_input_is_required"));
				}
				if(!strlen($new_settings['proxy_port']))
				{
					$isFormValid = false;
					$this->form->getItemByPostVar('proxy_port')->setAlert($lng->txt("msg_input_is_required"));
				}
			}

			if($isFormValid)
			{
				$this->setup->getClient()->write();
				ilUtil::sendSuccess($lng->txt("msg_obj_modified"), true);
				$this->form->setValuesByPost();
			}
		}
	}
}