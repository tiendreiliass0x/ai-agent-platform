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
            this.conversationId = null;
            this.visitorId = null;
            this.sessionToken = null;
            this.sessionTokenExpiresAt = null;
            this.sessionTokenProvider = null;
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
            if (!options.agentId) {
                console.error('YourAgent: agentId is required');
                return;
            }

            if (!options.sessionToken && typeof options.fetchSessionToken !== 'function') {
                console.error('YourAgent: provide either a sessionToken or a fetchSessionToken callback');
                return;
            }

            // Set configuration with defaults
            this.config = {
                agentId: options.agentId,
                apiBaseUrl: options.apiBaseUrl || 'https://api.yourdomain.com',
                sessionToken: options.sessionToken || null,
                fetchSessionToken: typeof options.fetchSessionToken === 'function' ? options.fetchSessionToken : null,

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
            this.visitorId = options.visitorId || this.generateVisitorId();
            this.sessionTokenProvider = this.config.fetchSessionToken;

            if (this.config.sessionToken) {
                this.setSessionToken(this.config.sessionToken, options.sessionTokenExpiresAt || null);
            }

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

        generateVisitorId() {
            return 'visitor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 8);
        }

        /**
         * Set the active session token and optional expiration.
         */
        setSessionToken(token, expires) {
            this.sessionToken = token || null;
            if (!token) {
                this.sessionTokenExpiresAt = null;
                return;
            }

            if (!expires) {
                this.sessionTokenExpiresAt = null;
            } else if (expires instanceof Date) {
                this.sessionTokenExpiresAt = expires.getTime();
            } else if (typeof expires === 'number') {
                // Treat numeric value as seconds from now
                this.sessionTokenExpiresAt = Date.now() + (expires * 1000);
            } else if (typeof expires === 'string') {
                const parsed = Date.parse(expires);
                this.sessionTokenExpiresAt = isNaN(parsed) ? null : parsed;
            } else if (typeof expires === 'object' && expires !== null) {
                const { expiresIn, expiresAt } = expires;
                if (expiresAt) {
                    const parsed = Date.parse(expiresAt);
                    this.sessionTokenExpiresAt = isNaN(parsed) ? null : parsed;
                } else if (expiresIn) {
                    this.sessionTokenExpiresAt = Date.now() + (Number(expiresIn) * 1000);
                } else {
                    this.sessionTokenExpiresAt = null;
                }
            } else {
                this.sessionTokenExpiresAt = null;
            }
        }

        /**
         * Ensure a valid session token is available (refresh if needed).
         */
        async ensureSessionToken(forceRefresh = false) {
            const now = Date.now();
            const safetyWindowMs = 5000; // Refresh slightly early

            if (!forceRefresh && this.sessionToken) {
                if (!this.sessionTokenExpiresAt || (this.sessionTokenExpiresAt - safetyWindowMs) > now) {
                    return this.sessionToken;
                }
            }

            if (typeof this.sessionTokenProvider === 'function') {
                const result = await this.sessionTokenProvider();

                if (!result) {
                    throw new Error('Session token provider returned empty result');
                }

                if (typeof result === 'string') {
                    this.setSessionToken(result, null);
                } else if (typeof result === 'object') {
                    this.setSessionToken(result.token, {
                        expiresIn: result.expiresIn,
                        expiresAt: result.expiresAt
                    });
                } else {
                    throw new Error('Unsupported session token format from provider');
                }

                if (!this.sessionToken) {
                    throw new Error('Session token provider did not supply a token');
                }

                return this.sessionToken;
            }

            if (!this.sessionToken) {
                throw new Error('Session token required but not available');
            }

            return this.sessionToken;
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
                const sessionContext = this.getUserContext();
                const summary = await this.streamChat(message, sessionContext);

                if (summary?.customer_context?.insights) {
                    this.updateUserProfile(summary.customer_context.insights);
                }

                if (summary?.customer_context?.escalation?.created) {
                    this.handleEscalation(summary.customer_context.escalation);
                }

                this.trackEvent('response_received', {
                    confidence: summary?.confidence_score || 0,
                    escalation: summary?.customer_context?.escalation?.created || false
                });

            } catch (error) {
                console.error('Error sending message:', error);

                this.addMessage(
                    "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
                    'assistant',
                    { error: true }
                );

                this.trackEvent('error', { error: error.message });
            }
        }

        async streamChat(message, sessionContext = {}) {
            const token = await this.ensureSessionToken();
            const payload = {
                message,
                conversation_id: this.conversationId,
                user_id: this.sessionId,
                session_context: sessionContext
            };

            const response = await fetch(`${this.apiBaseUrl}/api/v1/chat/${this.config.agentId}/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok || !response.body) {
                this.hideTyping();
                throw new Error(`Streaming request failed: ${response.status} ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            const assistantEntry = this.addMessage('', 'assistant');
            let placeholderActive = true;
            let aggregated = '';
            const summary = {
                confidence_score: 0,
                persona_applied: null,
                sources_count: 0,
                customer_context: {}
            };

            const removePlaceholder = () => {
                const index = this.conversationHistory.indexOf(assistantEntry.historyEntry);
                if (index >= 0) {
                    this.conversationHistory.splice(index, 1);
                }
                if (assistantEntry.element && assistantEntry.element.parentNode) {
                    assistantEntry.element.parentNode.removeChild(assistantEntry.element);
                }
            };

            const commitAssistantContent = () => {
                assistantEntry.historyEntry.content = aggregated;
                assistantEntry.historyEntry.metadata = summary;
            };

            const processEvent = (payload) => {
                switch (payload.type) {
                    case 'conversation':
                        if (payload.conversation_id) {
                            this.conversationId = payload.conversation_id;
                        }
                        break;
                    case 'metadata':
                        if (payload.sources) {
                            summary.sources_count = payload.sources.length;
                        }
                        break;
                    case 'content':
                        aggregated += payload.content || '';
                        if (aggregated.length > 0) {
                            placeholderActive = false;
                        }
                        assistantEntry.textElement.innerHTML = this.formatMessage(aggregated);
                        this.scrollToBottom();
                        break;
                    case 'done':
                        if (payload.conversation_id) {
                            this.conversationId = payload.conversation_id;
                        }
                        summary.confidence_score = payload.confidence_score || summary.confidence_score;
                        summary.persona_applied = payload.persona_applied || summary.persona_applied;
                        if (payload.sources_count != null) {
                            summary.sources_count = payload.sources_count;
                        }
                        if (payload.customer_context) {
                            summary.customer_context = payload.customer_context;
                        }
                        break;
                    case 'error':
                        throw new Error(payload.message || 'Streaming error');
                    default:
                        break;
                }
            };

            const processBuffer = () => {
                let boundary = buffer.indexOf('\n\n');
                while (boundary !== -1) {
                    const rawEvent = buffer.slice(0, boundary).trim();
                    buffer = buffer.slice(boundary + 2);

                    if (rawEvent.startsWith('data:')) {
                        const dataStr = rawEvent.slice(5).trim();
                        if (dataStr) {
                            let payload;
                            try {
                                payload = JSON.parse(dataStr);
                            } catch (err) {
                                console.warn('Invalid SSE payload', err);
                                boundary = buffer.indexOf('\n\n');
                                continue;
                            }

                            processEvent(payload);
                        }
                    }

                    boundary = buffer.indexOf('\n\n');
                }
            };

            try {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    processBuffer();
                }

                // Flush remaining buffer
                const remaining = decoder.decode();
                if (remaining) {
                    buffer += remaining;
                }
                processBuffer();

                commitAssistantContent();

            } catch (error) {
                removePlaceholder();
                throw error;
            } finally {
                this.hideTyping();
            }

            summary["text"] = aggregated;
            return summary;
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

            const textEl = messageEl.querySelector('.youragent-message-text');

            // Store in conversation history
            const historyEntry = {
                role: sender === 'user' ? 'user' : 'assistant',
                content: text,
                timestamp: new Date().toISOString(),
                metadata: metadata
            };
            this.conversationHistory.push(historyEntry);

            return {
                element: messageEl,
                textElement: textEl,
                historyEntry
            };
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
            const token = await this.ensureSessionToken();
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
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
