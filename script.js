/* ==========================================================================
   PuffPetals Interactive JavaScript Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initMobileNav();
    initProductTabs();
    initFloatingPetals();
    initSparkleClickEffect();
    initContactForm();
    initAddToCart();
});

/* ==========================================================================
   1. Mobile Navigation Menu Toggle
   ========================================================================== */
function initMobileNav() {
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            navMenu.classList.toggle('open');
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (navMenu.classList.contains('open') && !navMenu.contains(e.target) && e.target !== navToggle) {
                navMenu.classList.remove('open');
            }
        });

        // Close menu when navigation link is clicked
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('open');
                
                // Active link highlight
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            });
        });
    }
}

/* ==========================================================================
   2. Product Tabs Selection
   ========================================================================== */
function initProductTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const productGrids = document.querySelectorAll('.products-grid');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Switch active tab button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Switch active product grid
            productGrids.forEach(grid => {
                grid.classList.remove('active');
                if (grid.id === `tab-${targetTab}`) {
                    grid.classList.add('active');
                }
            });
        });
    });
}

/* ==========================================================================
   3. Programmatic Floating Petals Generator
   ========================================================================== */
function initFloatingPetals() {
    const container = document.getElementById('petal-container');
    if (!container) return;

    const petalCount = 15;
    const petalClasses = ['', 'petal-lavender', 'petal-green'];

    for (let i = 0; i < petalCount; i++) {
        createPetal(container, petalClasses);
    }
}

function createPetal(container, petalClasses) {
    const petal = document.createElement('div');
    const size = Math.random() * 15 + 10; // width/height between 10px and 25px
    const petalClass = petalClasses[Math.floor(Math.random() * petalClasses.length)];

    petal.classList.add('petal');
    if (petalClass) {
        petal.classList.add(petalClass);
    }

    petal.style.width = `${size}px`;
    petal.style.height = `${size}px`;
    petal.style.left = `${Math.random() * 100}vw`;
    
    // Set randomized falling speeds and delays
    const duration = Math.random() * 12 + 8; // 8s to 20s
    const delay = Math.random() * -20; // negative delay to spawn immediately across the screen height
    
    petal.style.animationDuration = `${duration}s`;
    petal.style.animationDelay = `${delay}s`;

    container.appendChild(petal);

    // Reset position after animation finishes to prevent pile-up
    petal.addEventListener('animationiteration', () => {
        petal.style.left = `${Math.random() * 100}vw`;
    });
}

/* ==========================================================================
   4. Interactive Sparkle Spark Particles on Click
   ========================================================================== */
function initSparkleClickEffect() {
    document.addEventListener('click', (e) => {
        // Skip sparkles for buttons, inputs, links to prevent clutter
        const ignoreTags = ['BUTTON', 'A', 'INPUT', 'TEXTAREA'];
        if (ignoreTags.includes(e.target.tagName) || e.target.closest('button') || e.target.closest('a')) {
            return;
        }

        const sparkle = document.createElement('div');
        sparkle.classList.add('sparkle-node');
        sparkle.innerHTML = '✨';
        sparkle.style.fontSize = `${Math.random() * 10 + 15}px`; // 15px to 25px
        sparkle.style.left = `${e.clientX - 10}px`;
        sparkle.style.top = `${e.clientY - 10}px`;

        document.body.appendChild(sparkle);

        // Remove element after animation completes
        setTimeout(() => {
            sparkle.remove();
        }, 1000);
    });
}

/* ==========================================================================
   5. Inquiry Form Submission Handling
   ========================================================================== */
function initContactForm() {
    const form = document.getElementById('contact-form');
    const successAlert = document.getElementById('form-success');

    if (form && successAlert) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            
            // Gather input values (demonstration capture)
            const name = document.getElementById('form-name').value;
            const email = document.getElementById('form-email').value;
            const phone = document.getElementById('form-phone').value;
            const message = document.getElementById('form-message').value;

            console.log('Inquiry Received:', { name, email, phone, message });

            // Animate transition to success block
            form.style.opacity = '0';
            setTimeout(() => {
                form.classList.add('hide');
                successAlert.classList.remove('hide');
                successAlert.style.opacity = '0';
                successAlert.style.transition = 'opacity 0.5s ease';
                setTimeout(() => {
                    successAlert.style.opacity = '1';
                }, 50);
            }, 300);

            // Pop cart toast confirmation
            showToast(`Thank you, ${name}! Your inquiry has been sent.`);
        });
    }
}

/* ==========================================================================
   6. Product Inquiry Selection Toast (Micro interaction)
   ========================================================================== */
let toastTimeout;
function showToast(message) {
    const toast = document.getElementById('cart-toast');
    const toastMsg = document.getElementById('toast-message');

    if (toast && toastMsg) {
        toastMsg.textContent = message;
        toast.classList.remove('hide');
        
        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toast.style.animation = 'slideInLeft 0.4s reverse forwards';
            setTimeout(() => {
                toast.classList.add('hide');
                toast.style.animation = ''; // Reset animation styles
            }, 400);
        }, 3000);
    }
}

function initAddToCart() {
    const addBtns = document.querySelectorAll('.add-to-cart-btn');
    addBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const productName = btn.getAttribute('data-name');
            const productPrice = btn.getAttribute('data-price');
            
            showToast(`Added ${productName} (₹${productPrice}) to inquiry list!`);
            
            // Fill details into contact form if user scrolls down
            const formMessage = document.getElementById('form-message');
            if (formMessage) {
                const currentVal = formMessage.value;
                if (!currentVal.includes(productName)) {
                    formMessage.value = currentVal 
                        ? `${currentVal}\n- ${productName} (₹${productPrice})`
                        : `I am interested in:\n- ${productName} (₹${productPrice})`;
                }
            }
        });
    });
}

/* ==========================================================================
   7. PuffBot Floating Chatbot Implementation
   ========================================================================== */
function toggleChatbot() {
    const chatbotContainer = document.getElementById('chatbot-container');
    if (chatbotContainer) {
        chatbotContainer.classList.toggle('active');
        
        // Auto scroll to bottom when opening chat
        if (chatbotContainer.classList.contains('active')) {
            setTimeout(scrollChatToBottom, 300);
        }
    }
}

// Global scope bindings for inline calls
window.toggleChatbot = toggleChatbot;
window.handleQuickReply = handleQuickReply;
window.handleChatEnter = handleChatEnter;
window.sendUserMessage = sendUserMessage;

/* Timestamps generator */
function getFormattedTime() {
    const now = new Date();
    let hours = now.getHours();
    let minutes = now.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12; // 12-hour clock
    minutes = minutes < 10 ? '0' + minutes : minutes;
    return `${hours}:${minutes} ${ampm}`;
}

/* Scroll logs container */
function scrollChatToBottom() {
    const chatBody = document.getElementById('chat-body');
    if (chatBody) {
        chatBody.scrollTo({
            top: chatBody.scrollHeight,
            behavior: 'smooth'
        });
    }
}

/* Append user/bot message bubble */
function appendMessage(sender, text) {
    const chatBody = document.getElementById('chat-body');
    if (!chatBody) return;

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', sender);

    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('message-bubble');
    bubbleDiv.textContent = text;

    const metaDiv = document.createElement('div');
    metaDiv.classList.add('message-meta');
    metaDiv.textContent = getFormattedTime();

    messageDiv.appendChild(bubbleDiv);
    messageDiv.appendChild(metaDiv);
    chatBody.appendChild(messageDiv);

    scrollChatToBottom();
}

/* Handle quick replies clicking */
function handleQuickReply(topic) {
    appendMessage('user', topic);
    processBotResponse(topic);
}

/* Handle Enter key press in chat input */
function handleChatEnter(event) {
    if (event.key === 'Enter') {
        sendUserMessage();
    }
}

/* Send text box message */
function sendUserMessage() {
    const input = document.getElementById('chat-input');
    if (!input) return;

    const text = input.value.trim();
    if (text === '') return;

    appendMessage('user', text);
    input.value = '';

    processBotResponse(text);
}

/* Match query against keyword dictionary & trigger replies with typing indicators */
function processBotResponse(query) {
    const normalizedQuery = query.toLowerCase().trim();
    let responseText = '';

    // 1. Bouquet Prices keyword search
    if (
        normalizedQuery.includes('price') || 
        normalizedQuery.includes('rate') || 
        normalizedQuery.includes('cost') || 
        normalizedQuery.includes('how much') || 
        normalizedQuery.includes('bouquet prices')
    ) {
        responseText = `Our bouquet prices are:

Mini Bouquets
• 3 Flowers – ₹199
• 5 Flowers – ₹299
• 7 Flowers – ₹399

Classic Bouquets
• 10 Flowers – ₹599
• 12 Flowers – ₹699
• 15 Flowers – ₹899

Premium Bouquets
• 20 Flowers – ₹1,199
• 25 Flowers – ₹1,499
• 30 Flowers – ₹1,799`;
    }
    // 2. Flower types keyword search
    else if (
        normalizedQuery.includes('what flowers') || 
        normalizedQuery.includes('flower types') || 
        normalizedQuery.includes('types of flower') || 
        normalizedQuery.includes('which flowers') || 
        normalizedQuery.includes('make flowers') ||
        normalizedQuery.includes('types')
    ) {
        responseText = `We create handmade pipe cleaner:
🌹 Roses
🌷 Tulips
🌼 Daisies
🌻 Sunflowers
🤍 Lilies

We also make custom bouquets.`;
    }
    // 3. Delivery info keyword search
    else if (
        normalizedQuery.includes('delivery') || 
        normalizedQuery.includes('shipping') || 
        normalizedQuery.includes('ship') || 
        normalizedQuery.includes('deliver')
    ) {
        responseText = `We offer local delivery and shipping depending on your location. Contact us for delivery charges.`;
    }
    // 4. Custom orders keyword search
    else if (
        normalizedQuery.includes('custom') || 
        normalizedQuery.includes('customize') || 
        normalizedQuery.includes('design') || 
        normalizedQuery.includes('choice') || 
        normalizedQuery.includes('card')
    ) {
        responseText = `You can customize:
• Flower types
• Bouquet size
• Colors
• Wrapping
• Personalized message card`;
    }
    // 5. Payment choices keyword search
    else if (
        normalizedQuery.includes('payment') || 
        normalizedQuery.includes('pay') || 
        normalizedQuery.includes('accept') || 
        normalizedQuery.includes('upi') || 
        normalizedQuery.includes('gpay') || 
        normalizedQuery.includes('phonepe')
    ) {
        responseText = `We accept UPI, Google Pay, PhonePe, Paytm, and bank transfer.`;
    }
    // 6. Greetings keyword search
    else if (
        normalizedQuery.includes('hello') || 
        normalizedQuery.includes('hi') || 
        normalizedQuery.includes('hey') || 
        normalizedQuery.includes('greetings')
    ) {
        responseText = `Hello! 🌸 Welcome to PuffPetals! How can I help you today?`;
    }
    // 7. Contact options keyword search
    else if (
        normalizedQuery.includes('contact') || 
        normalizedQuery.includes('whatsapp') || 
        normalizedQuery.includes('instagram') || 
        normalizedQuery.includes('phone number') || 
        normalizedQuery.includes('order') || 
        normalizedQuery.includes('buy')
    ) {
        responseText = `Please message us through Instagram or WhatsApp to place your order.`;
    }
    // 8. Default fallback response
    else {
        responseText = `I'm sorry, I don't have that information yet. Please contact PuffPetals directly for more assistance. 🌸`;
    }

    // Toggle typing indicator
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.classList.remove('hide');
        scrollChatToBottom();

        // Simulate network typing delay
        setTimeout(() => {
            typingIndicator.classList.add('hide');
            appendMessage('bot', responseText);
        }, Math.random() * 400 + 800); // 0.8s to 1.2s delay
    }
}
