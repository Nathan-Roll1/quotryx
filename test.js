const fs = require('fs');

const html = fs.readFileSync('docs/index.html', 'utf8');

// Extract the M object
const mMatch = html.match(/const M = (\{[\s\S]*?\});/);
if (!mMatch) throw new Error("Could not find M object");
const M = JSON.parse(mMatch[1]);

const NUMERIC = [
  ['bedrooms',           'Beds'],
  ['bathrooms',          'Baths'],
  ['size_interior_sqft', 'Sqft'],
  ['lot_size_sqft',      'Lot sqft'],
  ['year_built',         'Year'],
  ['parking_spaces',     'Parking'],
];

const FLAGS = [
  ['is_house',           'House'],
  ['is_condo',           'Condo'],
  ['has_garage',         'Garage'],
  ['has_basement',       'Basement'],
  ['basement_finished',  'Fin. bsmt'],
  ['has_cooling',        'Cooling'],
];

function predict(inputs) {
  const nums = NUMERIC.map(([name]) => {
    let v = inputs[name] !== undefined ? inputs[name] : '';
    let val = v === '' ? M.feature_means[name] : parseFloat(v);
    if (name === 'lot_size_sqft') {
      let idx = 0;
      while (idx < M.lot_sqfts_sorted.length && M.lot_sqfts_sorted[idx] < val) idx++;
      val = M.lot_sqfts_sorted.length > 0 ? (idx / M.lot_sqfts_sorted.length) * 100 : 50.0;
    }
    return val;
  });
  
  const flags = FLAGS.map(([name]) => inputs[name] ? 1 : 0);
  const base = nums.concat(flags);

  const poly = M.powers.map(row => {
    let term = 1;
    for (let j = 0; j < row.length; j++) {
      if (row[j] === 1) term *= base[j];
      else if (row[j] === 2) term *= base[j] * base[j];
    }
    return term;
  });

  const city = inputs.city || 'Prince Albert';
  const cityVec = [city === 'Saskatoon' ? 1 : 0, city === 'Regina' ? 1 : 0];
  const features = poly.concat(cityVec);

  let p = M.intercept;
  for (let i = 0; i < features.length; i++) p += M.coefficients[i] * features[i];

  return Math.max(p, 0);
}

const fmt = n => Math.round(n).toLocaleString('en-US');

// Test 1: Empty
console.log("Empty Form Prediction: $" + fmt(predict({})));

// Test 2: Specific values
let data = {
  "city": "Saskatoon",
  "bedrooms": 3,
  "bathrooms": 2,
  "size_interior_sqft": 1200,
  "lot_size_sqft": 5000,
  "year_built": 2000,
  "parking_spaces": 2,
  "is_house": 1,
  "is_condo": 0,
  "has_garage": 1,
  "has_basement": 1,
  "basement_finished": 1,
  "has_cooling": 1
};
console.log("Test 2 (Typical House) Prediction: $" + fmt(predict(data)));

// Test 3: Huge lot
data.lot_size_sqft = 1000000;
console.log("Test 3 (Huge Lot) Prediction: $" + fmt(predict(data)));

// Test 4: Tiny lot
data.lot_size_sqft = 10;
console.log("Test 4 (Tiny Lot) Prediction: $" + fmt(predict(data)));
