/**
 * AI Agent Complete Embeddable Widget
 * Production-ready widget with inline CSS and full customization
 * Version: 1.0.0
 */

(function(window, document) {
    'use strict';

    // Prevent multiple initializations
    if (window.YourAgent) {
        return;
    }

    // Inject CSS styles
    function injectStyles() {
        if (document.getElementById('youragent-styles')) {
            return; // Already injected
        }

        const style = document.createElement('style');
        style.id = 'youragent-styles';
        style.textContent = `
/* AI Agent Widget Styles */
:root{--youragent-primary-color:#2563eb;--youragent-accent-color:#3b82f6;--youragent-text-color:#1f2937;--youragent-bg-color:#ffffff;--youragent-border-color:#e5e7eb;--youragent-shadow:0 20px 25px -5px rgba(0,0,0,0.1),0 10px 10px -5px rgba(0,0,0,0.04);--youragent-shadow-lg:0 25px 50px -12px rgba(0,0,0,0.25);--youragent-border-radius:16px;--youragent-font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Roboto',sans-serif}.youragent-widget,.youragent-widget *{box-sizing:border-box;margin:0;padding:0}.youragent-widget{position:fixed;z-index:2147483647;font-family:var(--youragent-font-family);font-size:14px;line-height:1.4;color:var(--youragent-text-color);direction:ltr;text-align:left;pointer-events:none}.youragent-widget *{pointer-events:auto}.youragent-bottom-right{bottom:20px;right:20px}.youragent-bottom-left{bottom:20px;left:20px}.youragent-top-right{top:20px;right:20px}.youragent-top-left{top:20px;left:20px}.youragent-toggle{position:relative;width:60px;height:60px;background:var(--youragent-primary-color);border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:var(--youragent-shadow);transition:all 0.3s cubic-bezier(0.4,0,0.2,1);pointer-events:auto}.youragent-toggle:hover{transform:scale(1.05);box-shadow:var(--youragent-shadow-lg)}.youragent-toggle svg{width:24px;height:24px;color:white;transition:all 0.3s ease}.youragent-icon-close{position:absolute;opacity:0;transform:rotate(180deg)}.youragent-open .youragent-icon-chat{opacity:0;transform:rotate(-180deg)}.youragent-open .youragent-icon-close{opacity:1;transform:rotate(0deg)}.youragent-notification-badge{position:absolute;top:-2px;right:-2px;background:#ef4444;color:white;border-radius:50%;width:20px;height:20px;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;border:2px solid white}.youragent-chat{position:absolute;width:380px;height:600px;max-height:80vh;background:var(--youragent-bg-color);border-radius:var(--youragent-border-radius);box-shadow:var(--youragent-shadow-lg);display:flex;flex-direction:column;opacity:0;transform:scale(0.8) translateY(20px);transition:all 0.3s cubic-bezier(0.4,0,0.2,1);pointer-events:none;overflow:hidden}.youragent-bottom-right .youragent-chat,.youragent-top-right .youragent-chat{right:0}.youragent-bottom-left .youragent-chat,.youragent-top-left .youragent-chat{left:0}.youragent-bottom-right .youragent-chat,.youragent-bottom-left .youragent-chat{bottom:80px}.youragent-top-right .youragent-chat,.youragent-top-left .youragent-chat{top:80px}.youragent-open .youragent-chat{opacity:1;transform:scale(1) translateY(0);pointer-events:auto}.youragent-header{background:var(--youragent-primary-color);color:white;padding:16px 20px;display:flex;align-items:center;justify-content:space-between;border-radius:var(--youragent-border-radius) var(--youragent-border-radius) 0 0}.youragent-header-content{display:flex;align-items:center;gap:12px}.youragent-avatar{width:40px;height:40px;border-radius:50%;border:2px solid rgba(255,255,255,0.2)}.youragent-avatar-default{width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,0.2);display:flex;align-items:center;justify-content:center;font-size:20px}.youragent-header-text{flex:1}.youragent-title{font-weight:600;font-size:16px;margin-bottom:2px}.youragent-subtitle{font-size:13px;opacity:0.9}.youragent-minimize{background:none;border:none;color:white;cursor:pointer;padding:8px;border-radius:50%;display:flex;align-items:center;justify-content:center;transition:background-color 0.2s ease}.youragent-minimize:hover{background:rgba(255,255,255,0.1)}.youragent-minimize svg{width:18px;height:18px}.youragent-messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}.youragent-messages::-webkit-scrollbar{width:6px}.youragent-messages::-webkit-scrollbar-track{background:transparent}.youragent-messages::-webkit-scrollbar-thumb{background:rgba(0,0,0,0.1);border-radius:3px}.youragent-messages::-webkit-scrollbar-thumb:hover{background:rgba(0,0,0,0.2)}.youragent-message{display:flex;margin-bottom:8px}.youragent-message-user{justify-content:flex-end}.youragent-message-assistant{justify-content:flex-start}.youragent-message-content{max-width:80%;position:relative}.youragent-message-text{padding:12px 16px;border-radius:18px;position:relative;word-wrap:break-word;line-height:1.4}.youragent-message-user .youragent-message-text{background:var(--youragent-primary-color);color:white;border-bottom-right-radius:4px}.youragent-message-assistant .youragent-message-text{background:#f3f4f6;color:var(--youragent-text-color);border-bottom-left-radius:4px}.youragent-message-time{font-size:11px;color:#6b7280;margin-top:4px;text-align:center}.youragent-message-text strong{font-weight:600}.youragent-message-text em{font-style:italic}.youragent-message-text code{background:rgba(0,0,0,0.1);padding:2px 4px;border-radius:3px;font-family:'Monaco','Menlo','Ubuntu Mono',monospace;font-size:13px}.youragent-message-text a{color:var(--youragent-primary-color);text-decoration:underline}.youragent-message-user .youragent-message-text a{color:rgba(255,255,255,0.9)}.youragent-typing{padding:0 20px 16px;display:flex;align-items:center;gap:8px}.youragent-typing-indicator{display:flex;gap:3px}.youragent-typing-indicator span{width:6px;height:6px;border-radius:50%;background:#9ca3af;animation:youragent-typing-pulse 1.4s infinite ease-in-out}.youragent-typing-indicator span:nth-child(1){animation-delay:-0.32s}.youragent-typing-indicator span:nth-child(2){animation-delay:-0.16s}.youragent-typing-text{font-size:13px;color:#6b7280;margin-left:4px}@keyframes youragent-typing-pulse{0%,80%,100%{transform:scale(0.8);opacity:0.5}40%{transform:scale(1);opacity:1}}.youragent-input-container{border-top:1px solid var(--youragent-border-color);padding:16px 20px}.youragent-input-wrapper{display:flex;align-items:center;gap:8px;background:#f9fafb;border:1px solid var(--youragent-border-color);border-radius:24px;padding:8px 12px;transition:all 0.2s ease}.youragent-input-wrapper:focus-within{border-color:var(--youragent-primary-color);box-shadow:0 0 0 3px rgba(37,99,235,0.1)}.youragent-input{flex:1;border:none;background:none;outline:none;padding:8px 12px;font-size:14px;color:var(--youragent-text-color);resize:none;min-height:20px;max-height:100px;font-family:var(--youragent-font-family)}.youragent-input::placeholder{color:#9ca3af}.youragent-send{background:var(--youragent-primary-color);border:none;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all 0.2s ease}.youragent-send:hover{background:var(--youragent-accent-color);transform:scale(1.05)}.youragent-send svg{width:16px;height:16px;color:white}.youragent-powered-by{text-align:center;margin-top:8px;font-size:11px;color:#9ca3af}@media (max-width:480px){.youragent-mobile.youragent-widget{bottom:0!important;right:0!important;left:0!important;top:0!important}.youragent-mobile .youragent-toggle{bottom:20px;right:20px;position:fixed}.youragent-mobile .youragent-chat{position:fixed;top:0;left:0;right:0;bottom:0;width:100%!important;height:100%!important;max-height:100vh!important;border-radius:0!important;transform:translateY(100%)}.youragent-mobile.youragent-open .youragent-chat{transform:translateY(0)}.youragent-mobile .youragent-header{border-radius:0;padding:20px;padding-top:env(safe-area-inset-top,20px)}.youragent-mobile .youragent-messages{padding:16px}.youragent-mobile .youragent-input-container{padding:16px 20px;padding-bottom:env(safe-area-inset-bottom,16px)}.youragent-mobile.youragent-input-focused .youragent-chat{height:100vh}}.youragent-theme-dark{--youragent-bg-color:#1f2937;--youragent-text-color:#f9fafb;--youragent-border-color:#374151}.youragent-theme-dark .youragent-message-assistant .youragent-message-text{background:#374151;color:#f9fafb}.youragent-theme-dark .youragent-input-wrapper{background:#374151;border-color:#4b5563}.youragent-theme-dark .youragent-input{color:#f9fafb}.youragent-theme-dark .youragent-input::placeholder{color:#9ca3af}.youragent-theme-dark .youragent-powered-by{color:#6b7280}.youragent-theme-light{--youragent-bg-color:#ffffff;--youragent-text-color:#1f2937;--youragent-border-color:#e5e7eb}.youragent-fade-in{animation:youragent-fadeIn 0.3s ease}.youragent-slide-up{animation:youragent-slideUp 0.3s cubic-bezier(0.4,0,0.2,1)}@keyframes youragent-fadeIn{from{opacity:0}to{opacity:1}}@keyframes youragent-slideUp{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}.youragent-widget *:focus{outline:2px solid var(--youragent-primary-color);outline-offset:2px}.youragent-widget button:focus{outline-offset:-2px}@media (prefers-contrast:high){.youragent-widget{--youragent-border-color:#000000}.youragent-message-assistant .youragent-message-text{border:1px solid var(--youragent-border-color)}}@media (prefers-reduced-motion:reduce){.youragent-widget *{animation-duration:0.01ms!important;animation-iteration-count:1!important;transition-duration:0.01ms!important}}
        `;
        document.head.appendChild(style);
    }

    /**
     * Main Agent Widget Class
     */
    class AgentWidget {
        constructor() {
            this.initialized = false;
            this.config = {};
            this.sessionId = null;
            this.isOpen = false;
            this.isTyping = false;
            this.conversationHistory = [];
            this.userProfile = {};
            this.apiBaseUrl = '';

            // DOM elements
            this.container = null;
            this.chatWindow = null;
            this.chatMessages = null;
            this.chatInput = null;
            this.toggleButton = null;

            // Widget state
            this.connectionStatus = 'disconnected'; // disconnected, connecting, connected, error
            this.retryCount = 0;
            this.maxRetries = 3;

            // Bind methods
            this.init = this.init.bind(this);
            this.toggle = this.toggle.bind(this);
            this.sendMessage = this.sendMessage.bind(this);
            this.handleKeyPress = this.handleKeyPress.bind(this);
        }

        /**
         * Initialize the agent widget
         */
        init(options) {
            if (this.initialized) {
                console.warn('YourAgent already initialized');
                return;
            }

            // Inject CSS styles first
            injectStyles();

            // Validate required options
            if (!options.agentId || !options.apiKey) {
                console.error('YourAgent: agentId and apiKey are required');
                return;
            }

            // Set configuration with comprehensive defaults
            this.config = {
                agentId: options.agentId,
                apiKey: options.apiKey,
                apiBaseUrl: options.apiBaseUrl || 'https://api.yourdomain.com',

                // UI Configuration
                position: options.position || 'bottom-right',
                theme: options.theme || 'default',
                primaryColor: options.primaryColor || '#2563eb',
                accentColor: options.accentColor || '#3b82f6',
                borderRadius: options.borderRadius || '16px',

                // Widget Configuration
                welcomeMessage: options.welcomeMessage || "Hi! How can I help you today?",
                placeholder: options.placeholder || "Type your message...",
                title: options.title || "AI Assistant",
                subtitle: options.subtitle || "Powered by AI",
                avatar: options.avatar || null,

                // Behavior Configuration
                autoOpen: options.autoOpen || false,
                autoOpenDelay: options.autoOpenDelay || 1000,
                showTypingIndicator: options.showTypingIndicator !== false,
                showTimestamps: options.showTimestamps || false,
                enableFileUpload: options.enableFileUpload || false,
                enableEmojis: options.enableEmojis || true,
                enableSound: options.enableSound || false,

                // Advanced Configuration
                customCSS: options.customCSS || '',
                analytics: options.analytics !== false,
                debug: options.debug || false,
                retryOnError: options.retryOnError !== false,
                connectionTimeout: options.connectionTimeout || 10000,

                // Business Configuration
                businessHours: options.businessHours || null, // {start: '09:00', end: '17:00', timezone: 'UTC'}
                offlineMessage: options.offlineMessage || "We're currently offline. Leave a message and we'll get back to you!",
                escalationEnabled: options.escalationEnabled !== false,

                // Customization callbacks
                onOpen: options.onOpen || null,
                onClose: options.onClose || null,
                onMessage: options.onMessage || null,
                onError: options.onError || null,

                // Widget specific config from backend
                ...options.config
            };

            this.apiBaseUrl = this.config.apiBaseUrl;
            this.sessionId = this.generateSessionId();

            // Initialize widget
            this.createWidget();
            this.attachEventListeners();
            this.startSession();

            // Auto-open if configured
            if (this.config.autoOpen) {
                setTimeout(() => this.toggle(), this.config.autoOpenDelay);
            }

            this.initialized = true;

            if (this.config.debug) {
                console.log('YourAgent initialized:', this.config);
            }

            // Call initialization callback
            if (this.config.onOpen) {
                this.config.onOpen();
            }
        }

        /**
         * Generate unique session ID
         */
        generateSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        /**
         * Create the widget DOM structure
         */
        createWidget() {
            // Create main container
            this.container = document.createElement('div');
            this.container.id = 'youragent-widget';
            this.container.className = `youragent-widget youragent-${this.config.position} youragent-theme-${this.config.theme}`;

            // Create widget HTML
            this.container.innerHTML = `
                <!-- Chat Toggle Button -->
                <div id="youragent-toggle" class="youragent-toggle">
                    <div class="youragent-toggle-icon">
                        <svg class="youragent-icon-chat" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                        </svg>
                        <svg class="youragent-icon-close" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </div>
                    <div class="youragent-notification-badge" style="display: none;">1</div>
                </div>

                <!-- Chat Window -->
                <div id="youragent-chat" class="youragent-chat">
                    <!-- Chat Header -->
                    <div class="youragent-header">
                        <div class="youragent-header-content">
                            ${this.config.avatar ? `<img src="${this.config.avatar}" alt="Assistant" class="youragent-avatar">` : '<div class="youragent-avatar-default">🤖</div>'}
                            <div class="youragent-header-text">
                                <div class="youragent-title">${this.config.title}</div>
                                <div class="youragent-subtitle">${this.config.subtitle}</div>
                            </div>
                        </div>
                        <button class="youragent-minimize" id="youragent-minimize" title="Minimize chat">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <line x1="5" y1="12" x2="19" y2="12"/>
                            </svg>
                        </button>
                    </div>

                    <!-- Chat Messages -->
                    <div class="youragent-messages" id="youragent-messages">
                        ${this.getWelcomeMessage()}
                    </div>

                    <!-- Typing Indicator -->
                    <div class="youragent-typing" id="youragent-typing" style="display: none;">
                        <div class="youragent-typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        <div class="youragent-typing-text">Assistant is typing...</div>
                    </div>

                    <!-- Chat Input -->
                    <div class="youragent-input-container">
                        <div class="youragent-input-wrapper">
                            <input
                                type="text"
                                id="youragent-input"
                                class="youragent-input"
                                placeholder="${this.config.placeholder}"
                                autocomplete="off"
                                maxlength="1000"
                            >
                            <button class="youragent-send" id="youragent-send" title="Send message">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                    <line x1="22" y1="2" x2="11" y2="13"/>
                                    <polygon points="22,2 15,22 11,13 2,9 22,2"/>
                                </svg>
                            </button>
                        </div>
                        <div class="youragent-powered-by">
                            Powered by AI Agent Platform
                        </div>
                    </div>
                </div>
            `;

            // Add custom CSS if provided
            if (this.config.customCSS) {
                const styleEl = document.createElement('style');
                styleEl.textContent = this.config.customCSS;
                this.container.appendChild(styleEl);
            }

            // Append to body
            document.body.appendChild(this.container);

            // Store references to key elements
            this.toggleButton = document.getElementById('youragent-toggle');
            this.chatWindow = document.getElementById('youragent-chat');
            this.chatMessages = document.getElementById('youragent-messages');
            this.chatInput = document.getElementById('youragent-input');

            // Apply theme colors
            this.applyTheme();

            // Check for mobile
            this.checkMobile();
        }

        /**
         * Get welcome message HTML
         */
        getWelcomeMessage() {
            const timeString = this.config.showTimestamps ?
                `<div class="youragent-message-time">${this.formatTime(new Date())}</div>` : '';

            // Check business hours
            const isBusinessHours = this.checkBusinessHours();
            const message = isBusinessHours ?
                this.config.welcomeMessage :
                this.config.offlineMessage;

            return `
                <div class="youragent-message youragent-message-assistant youragent-fade-in">
                    <div class="youragent-message-content">
                        <div class="youragent-message-text">${message}</div>
                        ${timeString}
                    </div>
                </div>
            `;
        }

        /**
         * Check if within business hours
         */
        checkBusinessHours() {
            if (!this.config.businessHours) return true;

            const now = new Date();
            const hours = this.config.businessHours;

            // Simple check - could be enhanced for timezone support
            const currentHour = now.getHours();
            const startHour = parseInt(hours.start.split(':')[0]);
            const endHour = parseInt(hours.end.split(':')[0]);

            return currentHour >= startHour && currentHour < endHour;
        }

        /**
         * Apply theme and custom colors
         */
        applyTheme() {
            const root = this.container;
            root.style.setProperty('--youragent-primary-color', this.config.primaryColor);
            root.style.setProperty('--youragent-accent-color', this.config.accentColor);
            root.style.setProperty('--youragent-border-radius', this.config.borderRadius);
        }

        /**
         * Check if mobile and apply class
         */
        checkMobile() {
            if (window.innerWidth <= 480) {
                this.container.classList.add('youragent-mobile');
            }
        }

        /**
         * Attach event listeners
         */
        attachEventListeners() {
            // Toggle button click
            this.toggleButton.addEventListener('click', this.toggle);

            // Minimize button click
            const minimizeBtn = document.getElementById('youragent-minimize');
            minimizeBtn.addEventListener('click', this.toggle);

            // Send button click
            const sendBtn = document.getElementById('youragent-send');
            sendBtn.addEventListener('click', this.sendMessage);

            // Input keypress
            this.chatInput.addEventListener('keypress', this.handleKeyPress);

            // Input focus/blur for mobile optimization
            this.chatInput.addEventListener('focus', this.onInputFocus.bind(this));
            this.chatInput.addEventListener('blur', this.onInputBlur.bind(this));

            // Window resize for responsive behavior
            window.addEventListener('resize', this.onWindowResize.bind(this));

            // Prevent widget from being affected by page styles
            this.chatWindow.addEventListener('click', (e) => e.stopPropagation());

            // Keyboard accessibility
            this.toggleButton.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.toggle();
                }
            });

            // Add tabindex for accessibility
            this.toggleButton.setAttribute('tabindex', '0');
            this.toggleButton.setAttribute('role', 'button');
            this.toggleButton.setAttribute('aria-label', 'Open chat');
        }

        /**
         * Toggle chat window visibility
         */
        toggle() {
            this.isOpen = !this.isOpen;

            if (this.isOpen) {
                this.container.classList.add('youragent-open');
                setTimeout(() => {
                    if (this.chatInput) {
                        this.chatInput.focus();
                    }
                }, 300);
                this.trackEvent('widget_opened');
                this.toggleButton.setAttribute('aria-label', 'Close chat');

                // Callback
                if (this.config.onOpen) {
                    this.config.onOpen();
                }
            } else {
                this.container.classList.remove('youragent-open');
                this.trackEvent('widget_closed');
                this.toggleButton.setAttribute('aria-label', 'Open chat');

                // Callback
                if (this.config.onClose) {
                    this.config.onClose();
                }
            }
        }

        /**
         * Handle keypress events in input
         */
        handleKeyPress(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        }

        /**
         * Handle input focus (mobile optimization)
         */
        onInputFocus() {
            this.container.classList.add('youragent-input-focused');
        }

        /**
         * Handle input blur
         */
        onInputBlur() {
            this.container.classList.remove('youragent-input-focused');
        }

        /**
         * Handle window resize
         */
        onWindowResize() {
            this.checkMobile();
        }

        /**
         * Send a message
         */
        async sendMessage() {
            const message = this.chatInput.value.trim();
            if (!message) return;

            // Clear input
            this.chatInput.value = '';

            // Add user message to UI
            this.addMessage(message, 'user');

            // Show typing indicator
            this.showTyping();

            // Track user message
            this.trackEvent('message_sent', { message_length: message.length });

            // Call onMessage callback
            if (this.config.onMessage) {
                this.config.onMessage({ message, type: 'outgoing' });
            }

            try {
                this.connectionStatus = 'connecting';

                // Send to API with full context
                const response = await this.callAPI('/api/v1/chat/message', {
                    message: message,
                    sessionId: this.sessionId,
                    agentId: this.config.agentId,
                    contextData: this.getUserContext()
                });

                this.connectionStatus = 'connected';
                this.retryCount = 0;

                // Hide typing indicator
                this.hideTyping();

                // Add assistant response
                this.addMessage(response.response, 'assistant', response);

                // Update user profile if provided
                if (response.context_insights) {
                    this.updateUserProfile(response.context_insights);
                }

                // Handle escalation
                if (response.response_metadata?.escalation_recommended) {
                    this.handleEscalation(response.response_metadata);
                }

                // Play sound if enabled
                if (this.config.enableSound) {
                    this.playNotificationSound();
                }

                // Call onMessage callback
                if (this.config.onMessage) {
                    this.config.onMessage({ message: response.response, type: 'incoming' });
                }

                this.trackEvent('response_received', {
                    confidence: response.response_metadata?.confidence_score || 0,
                    escalation: response.response_metadata?.escalation_recommended || false
                });

            } catch (error) {
                this.hideTyping();
                this.connectionStatus = 'error';
                console.error('Error sending message:', error);

                let errorMessage = "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.";

                if (this.config.retryOnError && this.retryCount < this.maxRetries) {
                    this.retryCount++;
                    errorMessage += ` (Retry ${this.retryCount}/${this.maxRetries})`;

                    // Auto retry after delay
                    setTimeout(() => {
                        this.sendMessage();
                    }, 2000 * this.retryCount);
                }

                this.addMessage(errorMessage, 'assistant', { error: true });

                // Call error callback
                if (this.config.onError) {
                    this.config.onError(error);
                }

                this.trackEvent('error', { error: error.message, retry_count: this.retryCount });
            }
        }

        /**
         * Add a message to the chat
         */
        addMessage(text, sender, metadata = {}) {
            const messageEl = document.createElement('div');
            messageEl.className = `youragent-message youragent-message-${sender} youragent-slide-up`;

            const timestamp = this.config.showTimestamps ?
                `<div class="youragent-message-time">${this.formatTime(new Date())}</div>` : '';

            messageEl.innerHTML = `
                <div class="youragent-message-content">
                    <div class="youragent-message-text">${this.formatMessage(text)}</div>
                    ${timestamp}
                </div>
            `;

            this.chatMessages.appendChild(messageEl);
            this.scrollToBottom();

            // Store in conversation history
            this.conversationHistory.push({
                role: sender === 'user' ? 'user' : 'assistant',
                content: text,
                timestamp: new Date().toISOString(),
                metadata: metadata
            });

            // Limit conversation history
            if (this.conversationHistory.length > 50) {
                this.conversationHistory = this.conversationHistory.slice(-40);
            }
        }

        /**
         * Format message text (handle basic markdown, links, etc.)
         */
        formatMessage(text) {
            return text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>')
                .replace(/\n/g, '<br>');
        }

        /**
         * Format timestamp
         */
        formatTime(date) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }

        /**
         * Show typing indicator
         */
        showTyping() {
            if (this.config.showTypingIndicator) {
                this.isTyping = true;
                document.getElementById('youragent-typing').style.display = 'block';
                this.scrollToBottom();
            }
        }

        /**
         * Hide typing indicator
         */
        hideTyping() {
            this.isTyping = false;
            const typingEl = document.getElementById('youragent-typing');
            if (typingEl) {
                typingEl.style.display = 'none';
            }
        }

        /**
         * Scroll messages to bottom
         */
        scrollToBottom() {
            setTimeout(() => {
                if (this.chatMessages) {
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }
            }, 50);
        }

        /**
         * Play notification sound
         */
        playNotificationSound() {
            // Simple notification sound using Web Audio API
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();

                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);

                oscillator.frequency.value = 800;
                oscillator.type = 'sine';

                gainNode.gain.setValueAtTime(0, audioContext.currentTime);
                gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.01);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.5);
            } catch (e) {
                // Audio not supported, ignore
            }
        }

        /**
         * Get user context for API calls
         */
        getUserContext() {
            return {
                sessionId: this.sessionId,
                url: window.location.href,
                referrer: document.referrer,
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString(),
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                },
                conversationLength: this.conversationHistory.length,
                profile: this.userProfile,
                connectionStatus: this.connectionStatus,
                widgetVersion: '1.0.0'
            };
        }

        /**
         * Update user profile from API response
         */
        updateUserProfile(insights) {
            this.userProfile = {
                ...this.userProfile,
                ...insights,
                lastUpdated: new Date().toISOString()
            };
        }

        /**
         * Handle escalation scenarios
         */
        handleEscalation(metadata) {
            if (this.config.escalationEnabled) {
                // Could show special UI, offer contact options, etc.
                this.addMessage(
                    "I'd like to connect you with a human agent for better assistance. One moment please...",
                    'assistant',
                    { escalation: true }
                );
            }

            if (this.config.debug) {
                console.log('Escalation recommended:', metadata);
            }
        }

        /**
         * Start session with backend
         */
        async startSession() {
            try {
                await this.callAPI('/api/v1/sessions/' + this.sessionId + '/context', {
                    contextData: this.getUserContext()
                }, 'PUT');

                this.trackEvent('session_started');
            } catch (error) {
                if (this.config.debug) {
                    console.error('Error starting session:', error);
                }
            }
        }

        /**
         * Make API call to backend
         */
        async callAPI(endpoint, data = null, method = 'POST') {
            const url = `${this.apiBaseUrl}${endpoint}`;

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.connectionTimeout);

            try {
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': this.config.apiKey,
                        'X-Widget-Version': '1.0.0'
                    },
                    signal: controller.signal
                };

                if (data && (method === 'POST' || method === 'PUT')) {
                    options.body = JSON.stringify(data);
                }

                const response = await fetch(url, options);
                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`API call failed: ${response.status} ${response.statusText}`);
                }

                return await response.json();
            } catch (error) {
                clearTimeout(timeoutId);
                throw error;
            }
        }

        /**
         * Track analytics events
         */
        trackEvent(event, data = {}) {
            if (!this.config.analytics) return;

            const eventData = {
                event: event,
                agentId: this.config.agentId,
                sessionId: this.sessionId,
                timestamp: new Date().toISOString(),
                url: window.location.href,
                userAgent: navigator.userAgent,
                ...data
            };

            // Send to analytics endpoint (non-blocking)
            this.callAPI('/api/v1/analytics/events', eventData).catch(err => {
                if (this.config.debug) {
                    console.warn('Analytics tracking failed:', err);
                }
            });
        }

        /**
         * Public API methods
         */

        // Open the chat widget
        open() {
            if (!this.isOpen) {
                this.toggle();
            }
        }

        // Close the chat widget
        close() {
            if (this.isOpen) {
                this.toggle();
            }
        }

        // Send a programmatic message
        sendProgrammaticMessage(message) {
            if (typeof message === 'string' && message.trim()) {
                this.chatInput.value = message;
                this.sendMessage();
            }
        }

        // Update configuration
        updateConfig(newConfig) {
            this.config = { ...this.config, ...newConfig };
            this.applyTheme();
        }

        // Get current conversation
        getConversation() {
            return [...this.conversationHistory];
        }

        // Clear conversation
        clearConversation() {
            this.conversationHistory = [];
            this.chatMessages.innerHTML = this.getWelcomeMessage();
        }

        /**
         * Destroy the widget
         */
        destroy() {
            if (this.container && this.container.parentNode) {
                this.container.parentNode.removeChild(this.container);
            }

            // Remove styles if no other widgets exist
            const widgets = document.querySelectorAll('.youragent-widget');
            if (widgets.length === 0) {
                const styles = document.getElementById('youragent-styles');
                if (styles) {
                    styles.remove();
                }
            }

            this.initialized = false;
            this.trackEvent('widget_destroyed');
        }
    }

    // Create global instance
    window.YourAgent = new AgentWidget();

    // Auto-initialize if config is already available
    if (window.YourAgentConfig) {
        window.YourAgent.init(window.YourAgentConfig);
    }

})(window, document);