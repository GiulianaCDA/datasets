   <?php
   /**
    * @license http://www.mailcleaner.net/open/licence_en.html Mailcleaner Public License
    * @package mailcleaner
    * @author Olivier Diserens
    * @copyright 2009, Olivier Diserens
    */

   class UserController extends Zend_Controller_Action
   {
       public function loginAction()
       {
           $t = Zend_Registry::get('translate');
           $this->view->headTitle($t->_('login'));
           $this->view->layout()->setLayout('basic');

           $auth = Zend_Auth::getInstance();
           if ($auth->hasIdentity()) {
               $this->_redirect('/index');
           }

           $request = $this->getRequest();
           $redirect = $request->getPost('redirect', '/index'); // Set default value to '/index'
           $form = new Default_Form_Login();

           if ($this->getRequest()->isPost() && $form->isValid($request->getPost())) {
               $authAdapter = new Zend_Auth_Adapter_DbTable(
                   Zend_Registry::get('writedb'),
                   'administrator',
                   'username',
                   'password'
               );

               $givenUsername = $this->getRequest()->getParam('username');
               $givenPassword = $this->getRequest()->getParam('password');
               $authAdapter->setIdentity($givenUsername)->setCredential($givenPassword);

               try {
                   $result = $auth->authenticate($authAdapter);
                   if ($result->isValid()) {
                       $user = new Default_Model_Administrator();
                       $user->find($givenUsername);
                       Zend_Registry::set('user', $user);
                       $this->_redirect($redirect);
                   } else {
                       $form->addError('badCredentials');
                   }
               } catch (\Exception $e) {
                   // Handle exception appropriately (logging, error message, etc.)
                   $form->addError('internalServerError');
               }
           } else if ($this->getRequest()->isPost()) {
               $form->addError('badDataGiven');
           }

           // Rest of the code...
       }

       // Other actions...
   }
   