"""
Script complet pour tester le modèle CIFAR-10 en local
Assurez-vous d'avoir le fichier 'best_model.pth' dans le même dossier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 1. DÉFINITION DE L'ARCHITECTURE DU MODÈLE (identique à l'entraînement)
# ============================================================================

class CIFAR10Net(nn.Module):
    """Architecture CNN pour CIFAR-10"""
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        # Bloc 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Bloc 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Bloc 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Couches fully connected
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        # Bloc 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        
        # Bloc 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.bn2(x)
        
        # Bloc 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.bn3(x)
        
        # Flatten et FC
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================================

print("="*70)
print("CHARGEMENT DU MODÈLE CIFAR-10")
print("="*70)

# Détection du device (GPU ou CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé: {device}")

# Création et chargement du modèle
model = CIFAR10Net().to(device)

try:
    # Chargement du checkpoint
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Mode évaluation
    
    # Informations sur le modèle
    print(f"✓ Modèle chargé avec succès!")
    print(f"  - Epoch d'entraînement: {checkpoint['epoch'] + 1}")
    print(f"  - Précision sur test: {checkpoint['test_acc']:.2f}%")
    print(f"  - Loss sur test: {checkpoint['test_loss']:.4f}")
    
    # Classes CIFAR-10
    classes = checkpoint['classes']
    print(f"\nClasses disponibles: {', '.join(classes)}")
    
except FileNotFoundError:
    print("❌ Erreur: fichier 'best_model.pth' introuvable!")
    print("   Assurez-vous que le fichier est dans le même dossier que ce script.")
    exit()

print("="*70 + "\n")

# ============================================================================
# 3. TRANSFORMATION DES IMAGES
# ============================================================================

# Transformations à appliquer (identiques à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensionne à 32x32
    transforms.ToTensor(),  # Convertit en tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalisation CIFAR-10
])

# ============================================================================
# 4. FONCTION DE PRÉDICTION
# ============================================================================

def predict_image(image_path, show_image=True):
    """
    Prédit la classe d'une image
    
    Args:
        image_path: chemin vers l'image
        show_image: afficher l'image avec la prédiction
    
    Returns:
        classe_predite, confiance, probabilites
    """
    try:
        # Chargement de l'image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transformation
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Récupération des résultats
        predicted_class = classes[predicted.item()]
        confidence_value = confidence.item()
        all_probs = probabilities[0].cpu().numpy()
        
        # Affichage
        if show_image:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Image originale
            ax1.imshow(image)
            ax1.axis('off')
            ax1.set_title(f'Image originale ({original_size[0]}x{original_size[1]})', 
                         fontsize=12, fontweight='bold')
            
            # Graphique des probabilités
            sorted_indices = np.argsort(all_probs)[::-1][:5]  # Top 5
            top_classes = [classes[i] for i in sorted_indices]
            top_probs = [all_probs[i] * 100 for i in sorted_indices]
            
            colors = ['green' if i == 0 else 'lightblue' for i in range(5)]
            ax2.barh(top_classes, top_probs, color=colors)
            ax2.set_xlabel('Probabilité (%)', fontsize=11)
            ax2.set_title('Top 5 des prédictions', fontsize=12, fontweight='bold')
            ax2.invert_yaxis()
            
            # Ajout des valeurs sur les barres
            for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
                ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.show()
        
        return predicted_class, confidence_value, all_probs
        
    except FileNotFoundError:
        print(f"❌ Erreur: fichier '{image_path}' introuvable!")
        return None, None, None
    except Exception as e:
        print(f"❌ Erreur lors du traitement de l'image: {e}")
        return None, None, None

# ============================================================================
# 5. FONCTION POUR TESTER PLUSIEURS IMAGES
# ============================================================================

def predict_multiple_images(image_paths):
    """
    Prédit la classe de plusieurs images et affiche les résultats
    
    Args:
        image_paths: liste des chemins vers les images
    """
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx, image_path in enumerate(image_paths):
        try:
            # Chargement et prédiction
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)
            
            predicted_class = classes[predicted.item()]
            confidence_value = confidence.item()
            
            # Affichage
            ax = axes[idx] if n_images > 1 else axes[0]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'Prédiction: {predicted_class}\nConfiance: {confidence_value:.1%}',
                        fontsize=11, fontweight='bold',
                        color='green' if confidence_value > 0.7 else 'orange')
        except:
            ax = axes[idx] if n_images > 1 else axes[0]
            ax.text(0.5, 0.5, 'Erreur de chargement', ha='center', va='center')
            ax.axis('off')
    
    # Masquer les axes vides
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. TEST SUR LE DATASET CIFAR-10 (optionnel)
# ============================================================================

def test_on_cifar10_samples(n_samples=12):
    """
    Teste le modèle sur des échantillons aléatoires du test set CIFAR-10
    
    Args:
        n_samples: nombre d'images à tester
    """
    try:
        import torchvision
        from torch.utils.data import DataLoader
        
        print("Chargement du test set CIFAR-10...")
        
        # Transformation pour le test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True, 
            transform=transform_test
        )
        
        testloader = DataLoader(testset, batch_size=n_samples, shuffle=True)
        
        # Récupération d'un batch
        images, labels = next(iter(testloader))
        images, labels = images.to(device), labels.to(device)
        
        # Prédictions
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
        
        # Affichage
        cols = 4
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        axes = axes.flatten()
        
        for idx in range(n_samples):
            img = images[idx].cpu()
            # Dénormalisation
            img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            true_label = classes[labels[idx].item()]
            pred_label = classes[predicted[idx].item()]
            is_correct = labels[idx] == predicted[idx]
            
            # Affichage
            axes[idx].imshow(img.permute(1, 2, 0).numpy())
            axes[idx].axis('off')
            color = 'green' if is_correct else 'red'
            axes[idx].set_title(f'Vrai: {true_label}\nPréd: {pred_label}',
                               color=color, fontsize=10, fontweight='bold')
        
        # Masquer les axes vides
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Calcul de la précision sur ce batch
        correct = (predicted == labels).sum().item()
        accuracy = 100. * correct / n_samples
        print(f"\nPrécision sur ce batch: {accuracy:.2f}% ({correct}/{n_samples})")
        
    except Exception as e:
        print(f"Erreur lors du test sur CIFAR-10: {e}")

# ============================================================================
# 7. FONCTION POUR TESTER INTERACTIVEMENT
# ============================================================================

def test_interactive():
    """
    Mode interactif: demande le chemin de l'image à l'utilisateur
    """
    print("\n" + "="*70)
    print("MODE INTERACTIF - TEST D'IMAGES")
    print("="*70 + "\n")
    
    print("📝 Entrez le chemin de l'image à tester")
    print("   Exemples:")
    print("   - mon_chat.jpg")
    print("   - images/avion.png")
    print("   - C:/Users/Nom/Documents/chien.jpg")
    print("\n💡 Tapez 'q' ou 'quit' pour quitter\n")
    
    while True:
        print("-" * 70)
        image_path = input("📁 Chemin de l'image: ").strip()
        
        # Vérification pour quitter
        if image_path.lower() in ['q', 'quit', 'exit', 'quitter']:
            print("\n👋 Au revoir!")
            break
        
        # Vérification si le chemin est vide
        if not image_path:
            print("⚠️  Veuillez entrer un chemin valide\n")
            continue
        
        # Enlever les guillemets si présents (copier-coller)
        image_path = image_path.strip('"').strip("'")
        
        print("\n" + "="*70)
        
        # Test de l'image
        classe, confiance, probs = predict_image(image_path, show_image=True)
        
        if classe:
            print("\n" + "="*70)
            print("RÉSULTATS DÉTAILLÉS")
            print("="*70)
            print(f"\n🎯 PRÉDICTION: {classe.upper()}")
            print(f"📊 Confiance: {confiance:.2%}")
            
            print(f"\n📈 Probabilités pour toutes les classes:")
            print("-" * 50)
            
            # Trier par probabilité décroissante
            sorted_indices = np.argsort(probs)[::-1]
            
            for i, idx in enumerate(sorted_indices, 1):
                bar_length = int(probs[idx] * 40)  # Barre de 40 caractères max
                bar = "█" * bar_length
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"{emoji} {i:2d}. {classes[idx]:12s} {probs[idx]*100:6.2f}% {bar}")
            
            print("="*70)
        
        print("\n")

def load_images_from_folder(folder_path='.'):
    """
    Charge toutes les images d'un dossier
    
    Args:
        folder_path: chemin vers le dossier contenant les images (par défaut: dossier courant)
    
    Returns:
        liste des chemins d'images trouvées
    """
    import os
    
    # Extensions d'images supportées
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    image_paths = []
    
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(folder_path, file))
    
    return sorted(image_paths)

def test_local_images(folder_path='.', max_images=None):
    """
    Teste le modèle sur toutes les images d'un dossier local
    
    Args:
        folder_path: chemin vers le dossier (par défaut: dossier courant)
        max_images: nombre maximum d'images à tester (None = toutes)
    """
    import os
    
    print("\n" + "="*70)
    print(f"RECHERCHE D'IMAGES DANS: {os.path.abspath(folder_path)}")
    print("="*70 + "\n")
    
    # Chargement des images
    image_paths = load_images_from_folder(folder_path)
    
    if not image_paths:
        print("❌ Aucune image trouvée dans le dossier!")
        print("   Formats supportés: .jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp")
        print(f"\n💡 Placez vos images dans le dossier: {os.path.abspath(folder_path)}")
        return
    
    print(f"✓ {len(image_paths)} image(s) trouvée(s):")
    for i, path in enumerate(image_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")
    
    # Limiter le nombre d'images si spécifié
    if max_images and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        print(f"\n⚠️  Affichage limité aux {max_images} premières images")
    
    print("\n" + "-"*70)
    print("PRÉDICTIONS")
    print("-"*70 + "\n")
    
    # Test des images une par une avec détails
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {os.path.basename(image_path)}")
        print("-" * 50)
        
        classe, confiance, probs = predict_image(image_path, show_image=False)
        
        if classe:
            print(f"  🎯 Prédiction: {classe.upper()}")
            print(f"  📊 Confiance: {confiance:.2%}")
            print(f"\n  Top 3 des prédictions:")
            
            # Afficher le top 3
            top_indices = np.argsort(probs)[::-1][:3]
            for rank, idx in enumerate(top_indices, 1):
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                print(f"    {emoji} {classes[idx]:12s}: {probs[idx]*100:5.2f}%")
    
    # Affichage visuel de toutes les images
    print("\n" + "="*70)
    print("AFFICHAGE VISUEL DES RÉSULTATS")
    print("="*70 + "\n")
    
    predict_multiple_images(image_paths)
    
    print("\n✓ Test terminé!")

# ============================================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 TEST DU MODÈLE CIFAR-10")
    print("="*70 + "\n")
    
    print("Choisissez un mode de test:")
    print("  1. Mode interactif (tester des images une par une)")
    print("  2. Tester toutes les images d'un dossier")
    print("  3. Tester sur le dataset CIFAR-10")
    print("  4. Quitter")
    
    while True:
        print("\n" + "-"*70)
        choice = input("Votre choix (1/2/3/4): ").strip()
        
        if choice == '1':
            # Mode interactif
            test_interactive()
            break
            
        elif choice == '2':
            # Test d'un dossier
            print("\n📁 Entrez le chemin du dossier (ou appuyez sur Entrée pour le dossier courant):")
            folder = input("Chemin: ").strip()
            
            if not folder:
                folder = '.'
            
            folder = folder.strip('"').strip("'")
            
            print("\n📊 Nombre maximum d'images à afficher (ou appuyez sur Entrée pour toutes):")
            max_img = input("Nombre: ").strip()
            
            max_images = int(max_img) if max_img.isdigit() else None
            
            test_local_images(folder_path=folder, max_images=max_images)
            break
            
        elif choice == '3':
            # Test sur CIFAR-10
            print("\n📊 Nombre d'échantillons à tester (par défaut 12):")
            n_samples = input("Nombre: ").strip()
            
            n_samples = int(n_samples) if n_samples.isdigit() else 12
            
            test_on_cifar10_samples(n_samples=n_samples)
            break
            
        elif choice == '4':
            print("\n👋 Au revoir!")
            break
            
        else:
            print("⚠️  Choix invalide. Veuillez entrer 1, 2, 3 ou 4")
    
    print("\n" + "="*70)
    print("AUTRES FONCTIONS DISPONIBLES (utilisables dans le code):")
    print("="*70)
    print("\n• predict_image('image.jpg')           - Tester une image spécifique")
    print("• test_interactive()                   - Mode interactif")
    print("• test_local_images('dossier')         - Tester un dossier")
    print("• test_on_cifar10_samples(12)          - Tester sur CIFAR-10")
    print("\n" + "="*70 + "\n")