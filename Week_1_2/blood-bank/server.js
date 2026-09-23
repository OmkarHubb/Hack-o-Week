const express = require("express");
const path = require("path");
const donorRoutes = require("./routes/donors");

const app = express();

// Parse JSON request bodies
app.use(express.json());

// Serve static files (index.html, script.js) from the public folder
app.use(express.static(path.join(__dirname, "public")));

// Mount donor API routes
app.use("/api/donors", donorRoutes);

const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Blood Bank server running on http://localhost:${PORT}`);
});
