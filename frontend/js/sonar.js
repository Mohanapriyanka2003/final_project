/**
 * AI Underwater Object Detection System - Sonar Radar Screen
 * 
 * Draws the high-fidelity radial sonar sweep monitor on an HTML5 canvas.
 * Implements a rotating laser sweep line with a glowing phosphor decay trail,
 * concentric distance ring markings, compass degree ticks, and persistent glowing blips.
 */

class SonarRadar {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Settings
        this.maxDistanceMeters = 6.0;
        this.blips = [];            // Active echoing targets on screen: { angle, distance, class, alpha, timestamp }
        this.sweepHistory = [];     // Past sweep angles for rendering the trailing phosphor glow wedge
        this.maxHistorySize = 35;   // Trail length
        
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    /**
     * Spawns a glowing echo blip on the radar screen.
     */
    addBlip(angle, distance, targetClass, confidence) {
        if (!distance || distance < 0) return;
        
        // Check if a blip already exists at this exact coordinate neighborhood to avoid duplication
        const duplicateThreshold = 0.15; // meters
        const existing = this.blips.find(b => 
            Math.abs(b.angle - angle) < 4.0 && 
            Math.abs(b.distance - distance) < duplicateThreshold
        );
        
        if (existing) {
            existing.alpha = 1.0; // Refresh intensity glow
            existing.class = targetClass;
            existing.confidence = confidence;
            existing.timestamp = Date.now();
        } else {
            this.blips.push({
                angle: angle,
                distance: distance,
                class: targetClass,
                confidence: confidence,
                alpha: 1.0,
                timestamp: Date.now()
            });
        }
    }

    clearTrails() {
        this.blips = [];
        this.sweepHistory = [];
    }

    /**
     * Animates and renders the circular sonar sweep grid and glowing detections.
     */
    render(currentAngle, sweepForward) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Radar center rests at the bottom-center of the panel
        const cx = w / 2;
        const cy = h - 25;
        const radarRadius = Math.min(w / 2 - 20, h - 45);

        // Clear panel frame
        this.ctx.clearRect(0, 0, w, h);
        
        // 1. Maintain sweep trace history to draw a fading angular phosphor wedge
        this.sweepHistory.push(currentAngle);
        if (this.sweepHistory.length > this.maxHistorySize) {
            this.sweepHistory.shift();
        }

        // Draw radial background sweep wedge
        if (this.sweepHistory.length > 1) {
            for (let i = 0; i < this.sweepHistory.length - 1; i++) {
                const alpha = (i / this.sweepHistory.length) * 0.12;
                const a1 = (180.0 - this.sweepHistory[i]) * Math.PI / 180.0;
                const a2 = (180.0 - this.sweepHistory[i + 1]) * Math.PI / 180.0;
                
                this.ctx.fillStyle = `rgba(0, 240, 255, ${alpha})`;
                this.ctx.beginPath();
                this.ctx.moveTo(cx, cy);
                this.ctx.arc(cx, cy, radarRadius, Math.min(a1, a2), Math.max(a1, a2), false);
                this.ctx.closePath();
                this.ctx.fill();
            }
        }

        // 2. Draw Concentric Range Rings
        const ringsCount = 5;
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.08)';
        this.ctx.lineWidth = 1;
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = 'rgba(0, 240, 255, 0.4)';
        this.ctx.font = '7px var(--font-mono)';

        for (let i = 1; i <= ringsCount; i++) {
            const r = (i / ringsCount) * radarRadius;
            
            // Draw circle arc
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, r, Math.PI, 0, false);
            this.ctx.stroke();
            
            // Draw range label
            const distanceVal = (i / ringsCount) * this.maxDistanceMeters;
            this.ctx.fillText(`${distanceVal.toFixed(1)}m`, cx, cy - r + 9);
        }

        // 3. Draw Radial Compass Spoke Lines (every 30 degrees)
        for (let deg = 30; deg < 180; deg += 30) {
            const rad = (180.0 - deg) * Math.PI / 180.0;
            const rx = cx + radarRadius * Math.cos(rad);
            const ry = cy + radarRadius * Math.sin(rad);
            
            this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
            this.ctx.beginPath();
            this.ctx.moveTo(cx, cy);
            this.ctx.lineTo(rx, ry);
            this.ctx.stroke();
            
            // Draw angle text tags
            this.ctx.save();
            this.ctx.translate(rx + 10 * Math.cos(rad), ry + 10 * Math.sin(rad));
            this.ctx.font = '8px var(--font-mono)';
            this.ctx.fillText(`${deg}°`, 0, 0);
            this.ctx.restore();
        }
        
        // 4. Render Echo Blips (Phospor fading)
        const activeTimeThreshold = 18000; // Blips disappear after 18 seconds of no scans
        const now = Date.now();
        this.blips = this.blips.filter(b => (now - b.timestamp) < activeTimeThreshold);

        for (let b of this.blips) {
            // Decay alpha based on time elapsed
            const elapsed = now - b.timestamp;
            b.alpha = Math.max(0.04, 1.0 - (elapsed / activeTimeThreshold));
            
            // Convert polar to Cartesian relative to radar center
            const rad = (180.0 - b.angle) * Math.PI / 180.0;
            const rScaled = (b.distance / this.maxDistanceMeters) * radarRadius;
            const bx = cx + rScaled * Math.cos(rad);
            const by = cy + rScaled * Math.sin(rad);
            
            // Target Color Code
            let color = 'rgba(0, 136, 255, ';
            if (b.class === "Fish") color = 'rgba(0, 230, 118, ';
            if (b.class === "Submarine") color = 'rgba(255, 51, 85, ';
            
            // Glow drop shadow
            this.ctx.shadowBlur = 12 * b.alpha;
            this.ctx.shadowColor = color + b.alpha + ')';

            // Draw glowing core blip
            this.ctx.fillStyle = color + b.alpha + ')';
            this.ctx.beginPath();
            this.ctx.arc(bx, by, 6, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Dotted target sweep locator ring
            this.ctx.shadowBlur = 0;
            this.ctx.strokeStyle = color + (b.alpha * 0.4) + ')';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.arc(bx, by, 10 + (1.0 - b.alpha) * 12, 0, Math.PI * 2);
            this.ctx.stroke();

            // Label text tag (Fades out slower)
            if (b.alpha > 0.15) {
                this.ctx.fillStyle = `rgba(255, 255, 255, ${b.alpha * 0.85})`;
                this.ctx.font = 'bold 7px var(--font-primary)';
                this.ctx.fillText(b.class.toUpperCase(), bx, by - 9);
            }
        }

        // 5. Draw active sweeping laser line
        const sweepRad = (180.0 - currentAngle) * Math.PI / 180.0;
        const sweepX = cx + radarRadius * Math.cos(sweepRad);
        const sweepY = cy + radarRadius * Math.sin(sweepRad);
        
        // Laser glow
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = 'rgba(0, 240, 255, 0.8)';
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.95)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(cx, cy);
        this.ctx.lineTo(sweepX, sweepY);
        this.ctx.stroke();
        
        this.ctx.shadowBlur = 0; // reset
        
        // 6. Sonar Boundary outer dial ring
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
        this.ctx.lineWidth = 2.5;
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, radarRadius + 2, Math.PI, 0, false);
        this.ctx.stroke();
        
        // Dial border hatch markings
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
        this.ctx.lineWidth = 1;
        for (let a = 0; a <= 180; a += 5) {
            const rVal = (180 - a) * Math.PI / 180;
            const startX = cx + (radarRadius + 2) * Math.cos(rVal);
            const startY = cy + (radarRadius + 2) * Math.sin(rVal);
            const length = (a % 15 === 0) ? 7 : 3;
            const endX = cx + (radarRadius + 2 + length) * Math.cos(rVal);
            const endY = cy + (radarRadius + 2 + length) * Math.sin(rVal);
            
            this.ctx.beginPath();
            this.ctx.moveTo(startX, startY);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
        }
    }
}
