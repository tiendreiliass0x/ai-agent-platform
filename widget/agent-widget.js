/**
 * AI Agent Embeddable Widget
 * Production-ready widget for embedding world-class AI concierges on any website
 *
 * Features:
 * - Responsive design for all devices
 * - Customizable branding and styling
 * - Real-time chat with world-class intelligence
 * - Session management and user profiling
 * - Analytics and interaction tracking
 */

(function(window, document) {
    'use strict';

    // Prevent multiple initializations
    if (window.YourAgent) {
        return;
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

            // Bind methods
            this.init = this.init.bind(this);
            this.toggle = this.toggle.bind(this);
            this.sendMessage = this.sendMessage.bind(this);
            this.handleKeyPress = this.handleKeyPress.bind(this);
        }

        /**
         * Initialize the agent widget
         * @param {Object} options - Configuration options
         */
        init(options) {
            if (this.initialized) {
                console.warn('YourAgent already initialized');
                return;
            }

            // Validate required options
            if (!options.agentId || !options.apiKey) {
                console.error('YourAgent: agentId and apiKey are required');
                return;
            }

            // Set configuration with defaults
            this.config = {
                agentId: options.agentId,
                apiKey: options.apiKey,
                apiBaseUrl: options.apiBaseUrl || 'https://api.yourdomain.com',

                // UI Configuration
                position: options.position || 'bottom-right', // bottom-right, bottom-left, top-right, top-left
                theme: options.theme || 'default', // default, dark, light, custom
                primaryColor: options.primaryColor || '#2563eb',
                accentColor: options.accentColor || '#3b82f6',

                // Widget Configuration
                welcomeMessage: options.welcomeMessage || "Hi! How can I help you today?",
                placeholder: options.placeholder || "Type your message...",
                title: options.title || "AI Assistant",
                subtitle: options.subtitle || "Powered by AI",
                avatar: options.avatar || null,

                // Behavior Configuration
                autoOpen: options.autoOpen || false,
                showTypingIndicator: options.showTypingIndicator !== false,
                showTimestamps: options.showTimestamps || false,
                enableFileUpload: options.enableFileUpload || false,
                enableEmojis: options.enableEmojis || true,

                // Advanced Configuration
                customCSS: options.customCSS || '',
                analytics: options.analytics !== false,
                debug: options.debug || false,

                // Widget specific config from backend
                ...options.config
            };

            this.apiBaseUrl = this.config.apiBaseUrl;
            this.sessionId = this.generateSessionId();

            // Initialize widget
            this.createWidget();
            this.attachEventListeners();
            this.startSession();

            if (this.config.autoOpen) {
                setTimeout(() => this.toggle(), 1000);
            }

            this.initialized = true;

            if (this.config.debug) {
                console.log('YourAgent initialized:', this.config);
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
                        <button class="youragent-minimize" id="youragent-minimize">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <line x1="5" y1="12" x2="19" y2="12"/>
                            </svg>
                        </button>
                    </div>

                    <!-- Chat Messages -->
                    <div class="youragent-messages" id="youragent-messages">
                        <div class="youragent-message youragent-message-assistant">
                            <div class="youragent-message-content">
                                <div class="youragent-message-text">${this.config.welcomeMessage}</div>
                                ${this.config.showTimestamps ? '<div class="youragent-message-time">' + this.formatTime(new Date()) + '</div>' : ''}
                            </div>
                        </div>
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
                            >
                            <button class="youragent-send" id="youragent-send">
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
        }

        /**
         * Apply theme and custom colors
         */
        applyTheme() {
            const root = this.container;
            root.style.setProperty('--youragent-primary-color', this.config.primaryColor);
            root.style.setProperty('--youragent-accent-color', this.config.accentColor);
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
        }

        /**
         * Toggle chat window visibility
         */
        toggle() {
            this.isOpen = !this.isOpen;

            if (this.isOpen) {
                this.container.classList.add('youragent-open');
                this.chatInput.focus();
                this.trackEvent('widget_opened');
            } else {
                this.container.classList.remove('youragent-open');
                this.trackEvent('widget_closed');
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
            // Add class for mobile keyboard adjustments
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
            // Adjust widget size for mobile
            if (window.innerWidth <= 480) {
                this.container.classList.add('youragent-mobile');
            } else {
                this.container.classList.remove('youragent-mobile');
            }
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

            try {
                // Send to API with full context
                const response = await this.callAPI('/api/v1/chat/message', {
                    message: message,
                    sessionId: this.sessionId,
                    agentId: this.config.agentId,
                    contextData: this.getUserContext()
                });

                // Hide typing indicator
                this.hideTyping();

                // Add assistant response
                this.addMessage(response.response, 'assistant', response);

                // Update user profile if provided
                if (response.context_insights) {
                    this.updateUserProfile(response.context_insights);
                }

                // Handle any special actions
                if (response.response_metadata && response.response_metadata.escalation_recommended) {
                    this.handleEscalation(response.response_metadata);
                }

                this.trackEvent('response_received', {
                    confidence: response.response_metadata?.confidence_score || 0,
                    escalation: response.response_metadata?.escalation_recommended || false
                });

            } catch (error) {
                this.hideTyping();
                console.error('Error sending message:', error);

                this.addMessage(
                    "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
                    'assistant',
                    { error: true }
                );

                this.trackEvent('error', { error: error.message });
            }
        }

        /**
         * Add a message to the chat
         */
        addMessage(text, sender, metadata = {}) {
            const messageEl = document.createElement('div');
            messageEl.className = `youragent-message youragent-message-${sender}`;

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
        }

        /**
         * Format message text (handle basic markdown, links, etc.)
         */
        formatMessage(text) {
            return text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>')
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
            document.getElementById('youragent-typing').style.display = 'none';
        }

        /**
         * Scroll messages to bottom
         */
        scrollToBottom() {
            setTimeout(() => {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }, 50);
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
                profile: this.userProfile
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
            // Could show special UI, offer contact options, etc.
            if (this.config.debug) {
                console.log('Escalation recommended:', metadata);
            }
        }

        /**
         * Start session with backend
         */
        async startSession() {
            try {
                // Initialize session context
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
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': this.config.apiKey
                }
            };

            if (data && (method === 'POST' || method === 'PUT')) {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(url, options);

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }

            return await response.json();
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
         * Destroy the widget
         */
        destroy() {
            if (this.container && this.container.parentNode) {
                this.container.parentNode.removeChild(this.container);
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