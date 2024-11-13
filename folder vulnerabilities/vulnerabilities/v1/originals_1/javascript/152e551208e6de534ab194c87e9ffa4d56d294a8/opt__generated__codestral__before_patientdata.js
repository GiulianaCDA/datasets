 // Define a function to handle the form submission event
function handleFormSubmit(event) {
  event.preventDefault(); // Prevent default form submission behavior

  // Get the values from the form fields
  const mothersname = document.getElementById('mothersname').value;
  const guardiansname = document.getElementById('guardiansname').value;
  const allowImmRegUse = document.querySelector('input[name=allowImmRegUse]:checked').value;
  const allowImmInfoShare = document.querySelector('input[name=allowImmInfoShare]:checked').value;
  const allowHealthInfoEx = document.querySelector('input[name=allowHealthInfoEx]:checked').value;
  const allowPatientPortal = document.querySelector('input[name=allowPatientPortal]:checked').value;
  const careTeam = document.getElementById('careTeam').value || 0;
  const county = document.getElementById('county').value;
  const note = document.getElementById('note').value;

  // Create an object to hold the form data
  const formData = {
    mothersname,
    guardiansname,
    allowImmRegUse,
    allowImmInfoShare,
    allowHealthInfoEx,
    allowPatientPortal,
    careTeam,
    county,
    note,
  };

  // Send the form data to the server for processing using AJAX or fetch API
  // Example using fetch API:
  fetch('/api/update-patient', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(formData),
  })
    .then((response) => response.json())
    .then((data) => {
      // Handle the server's response here, e.g., show a success message or handle errors
    })
    .catch((error) => {
      console.error('Error:', error);
    });
}