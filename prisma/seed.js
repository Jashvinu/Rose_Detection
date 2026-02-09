// PlantVillage Rose Edition - Database Seeder
const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function main() {
  console.log('🌹 Seeding PlantVillage database...');

  // Create diseases
  const diseases = await Promise.all([
    prisma.disease.upsert({
      where: { name: 'Black Spot' },
      update: {},
      create: {
        name: 'Black Spot',
        scientificName: 'Diplocarpon rosae',
        description: 'Fungal disease causing black spots on rose leaves',
        symptoms: ['Black circular spots', 'Yellow halos around spots', 'Premature leaf drop'],
        treatments: ['Remove infected leaves', 'Apply fungicide', 'Improve air circulation'],
        severity: 'MODERATE',
      },
    }),
    prisma.disease.upsert({
      where: { name: 'Powdery Mildew' },
      update: {},
      create: {
        name: 'Powdery Mildew',
        scientificName: 'Podosphaera pannosa',
        description: 'Fungal disease causing white powdery coating on leaves',
        symptoms: ['White powdery coating', 'Distorted growth', 'Curled leaves'],
        treatments: ['Improve air circulation', 'Apply sulfur-based fungicide', 'Remove infected parts'],
        severity: 'MODERATE',
      },
    }),
    prisma.disease.upsert({
      where: { name: 'Rose Rust' },
      update: {},
      create: {
        name: 'Rose Rust',
        scientificName: 'Phragmidium tuberculatum',
        description: 'Fungal disease causing orange rust spots on leaves',
        symptoms: ['Orange pustules on leaves', 'Yellow spots on upper leaf surface', 'Leaf drop'],
        treatments: ['Remove infected leaves', 'Apply fungicide', 'Avoid overhead watering'],
        severity: 'HIGH',
      },
    }),
    prisma.disease.upsert({
      where: { name: 'Downy Mildew' },
      update: {},
      create: {
        name: 'Downy Mildew',
        scientificName: 'Peronospora sparsa',
        description: 'Pathogenic water mold causing purple-red blotches',
        symptoms: ['Purple-red irregular blotches', 'Gray fuzzy growth', 'Rapid defoliation'],
        treatments: ['Remove infected tissue', 'Improve ventilation', 'Apply phosphorous acid'],
        severity: 'CRITICAL',
      },
    }),
  ]);
  console.log(`✅ Created ${diseases.length} diseases`);

  // Create pests
  const pests = await Promise.all([
    prisma.pest.upsert({
      where: { name: 'Two-Spotted Spider Mite' },
      update: {},
      create: {
        name: 'Two-Spotted Spider Mite',
        scientificName: 'Tetranychus urticae',
        description: 'Tiny arachnid pest that feeds on plant cells',
        indicators: ['Fine webbing', 'Stippled leaves', 'Bronze/yellow discoloration'],
        treatments: ['Miticide application', 'Predatory mites', 'Neem oil spray'],
      },
    }),
    prisma.pest.upsert({
      where: { name: 'Rose Aphid' },
      update: {},
      create: {
        name: 'Rose Aphid',
        scientificName: 'Macrosiphum rosae',
        description: 'Small soft-bodied insects that suck plant sap',
        indicators: ['Clustered on new growth', 'Sticky honeydew', 'Distorted buds'],
        treatments: ['Insecticidal soap', 'Ladybugs', 'Strong water spray'],
      },
    }),
    prisma.pest.upsert({
      where: { name: 'Thrips' },
      update: {},
      create: {
        name: 'Thrips',
        scientificName: 'Frankliniella occidentalis',
        description: 'Tiny insects causing flower damage',
        indicators: ['Distorted petals', 'Brown streaks', 'Silver-gray scarring'],
        treatments: ['Blue sticky traps', 'Spinosad spray', 'Remove damaged flowers'],
      },
    }),
  ]);
  console.log(`✅ Created ${pests.length} pests`);

  // Create initial ML models reference
  const models = await Promise.all([
    prisma.mLModel.upsert({
      where: { name_version: { name: 'disease_detector', version: 'v1.0' } },
      update: {},
      create: {
        name: 'disease_detector',
        version: 'v1.0',
        mode: 'MODE_A_DISEASE',
        filePath: 'models/disease_detector_v1.pt',
        accuracy: 0.92,
        parameters: { epochs: 100, batchSize: 16, imageSize: 640 },
        isActive: true,
        trainedAt: new Date(),
      },
    }),
    prisma.mLModel.upsert({
      where: { name_version: { name: 'mite_counter', version: 'v1.0' } },
      update: {},
      create: {
        name: 'mite_counter',
        version: 'v1.0',
        mode: 'MODE_B_MITE',
        filePath: 'models/mite_counter_v1.pt',
        accuracy: 0.88,
        parameters: { epochs: 100, batchSize: 8, imageSize: 1280 },
        isActive: true,
        trainedAt: new Date(),
      },
    }),
  ]);
  console.log(`✅ Created ${models.length} ML model references`);

  console.log('\n🌹 PlantVillage database seeding complete!');
}

main()
  .catch((e) => {
    console.error('❌ Seeding error:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
