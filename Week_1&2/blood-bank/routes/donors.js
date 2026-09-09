const express = require("express");
const router = express.Router();
const donors = require("../data");

// GET /api/donors — return all donors
router.get("/", (req, res) => {
  res.status(200).json(donors);
});

// GET /api/donors/:id — return one donor
router.get("/:id", (req, res) => {
  const id = Number(req.params.id);
  const donor = donors.find((d) => d.id === id);

  if (!donor) {
    return res.status(404).json({ message: "Donor not found" });
  }

  res.status(200).json(donor);
});

// POST /api/donors — add a new donor
router.post("/", (req, res) => {
  const newDonor = {
    id: Date.now(),
    name: req.body.name,
    age: req.body.age,
    bloodGroup: req.body.bloodGroup,
    phone: req.body.phone,
    city: req.body.city,
    unitsAvailable: req.body.unitsAvailable,
  };

  donors.push(newDonor);
  res.status(201).json(newDonor);
});

// PUT /api/donors/:id — update an existing donor
router.put("/:id", (req, res) => {
  const id = Number(req.params.id);
  const donor = donors.find((d) => d.id === id);

  if (!donor) {
    return res.status(404).json({ message: "Donor not found" });
  }

  donor.name = req.body.name;
  donor.age = req.body.age;
  donor.bloodGroup = req.body.bloodGroup;
  donor.phone = req.body.phone;
  donor.city = req.body.city;
  donor.unitsAvailable = req.body.unitsAvailable;

  res.status(200).json(donor);
});

// DELETE /api/donors/:id — remove a donor
router.delete("/:id", (req, res) => {
  const id = Number(req.params.id);
  const index = donors.findIndex((d) => d.id === id);

  if (index === -1) {
    return res.status(404).json({ message: "Donor not found" });
  }

  const deleted = donors.splice(index, 1);
  res.status(200).json(deleted[0]);
});

module.exports = router;
