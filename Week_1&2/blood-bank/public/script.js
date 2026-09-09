const API = "/api/donors";

// Track whether we are editing an existing donor
let editingId = null;

// --- Fetch and display all donors ---
async function loadDonors() {
  const res = await fetch(API);
  const donors = await res.json();
  const tbody = document.getElementById("donorTable");

  tbody.innerHTML = donors
    .map(
      (d) => `
    <tr>
      <td>${d.name}</td>
      <td>${d.age}</td>
      <td>${d.bloodGroup}</td>
      <td>${d.phone}</td>
      <td>${d.city}</td>
      <td>${d.unitsAvailable}</td>
      <td>
        <button class="btn-edit" onclick="editDonor(${d.id})">Edit</button>
        <button class="btn-delete" onclick="deleteDonor(${d.id})">Delete</button>
      </td>
    </tr>`
    )
    .join("");
}

// --- Handle form submit (Add or Update) ---
document.getElementById("donorForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const donor = {
    name: document.getElementById("name").value,
    age: Number(document.getElementById("age").value),
    bloodGroup: document.getElementById("bloodGroup").value,
    phone: document.getElementById("phone").value,
    city: document.getElementById("city").value,
    unitsAvailable: Number(document.getElementById("unitsAvailable").value),
  };

  if (editingId) {
    // Update existing donor
    await fetch(`${API}/${editingId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(donor),
    });
    cancelEdit();
  } else {
    // Create new donor
    await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(donor),
    });
  }

  document.getElementById("donorForm").reset();
  loadDonors();
});

// --- Populate form for editing ---
async function editDonor(id) {
  const res = await fetch(`${API}/${id}`);
  const d = await res.json();

  document.getElementById("donorId").value = d.id;
  document.getElementById("name").value = d.name;
  document.getElementById("age").value = d.age;
  document.getElementById("bloodGroup").value = d.bloodGroup;
  document.getElementById("phone").value = d.phone;
  document.getElementById("city").value = d.city;
  document.getElementById("unitsAvailable").value = d.unitsAvailable;

  editingId = d.id;
  document.getElementById("submitBtn").textContent = "Update Donor";
  document.getElementById("cancelBtn").style.display = "inline";
}

// --- Cancel editing mode ---
function cancelEdit() {
  editingId = null;
  document.getElementById("donorForm").reset();
  document.getElementById("submitBtn").textContent = "Add Donor";
  document.getElementById("cancelBtn").style.display = "none";
}

// --- Delete a donor ---
async function deleteDonor(id) {
  await fetch(`${API}/${id}`, { method: "DELETE" });
  loadDonors();
}

// --- Load donors when page opens ---
loadDonors();
