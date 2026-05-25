/**
 * AI Underwater Object Detection System - Ocean Environment Sandbox
 * 
 * Implements a 2D canvas simulation of the underwater environment.
 * Users can interactively place, drag, and drop target objects (Submarine, Rock, Fish Shoal).
 * The environment renders fluid current particles and calculates the intersection between
 * the active sonar sweep cone and the target boundaries to supply real-time sensor echoes.
 */

class OceanEnvironment {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // List of target objects placed in the sandbox
        // Pos coordinates are relative to the canvas grid (0 to 100)
        // Sonar transducer resides at the bottom-center: (50, 95)
        this.sonarPos = { x: 50, y: 95 };
        this.targets = [
            { id: "rock_1", name: "Volcanic Rock Bed", x: 25, y: 35, type: "Rock", radius: 5.5, angle: 0, dist: 0 },
            { id: "fish_1", name: "Deep Sea Salmon Shoal", x: 50, y: 22, type: "Fish", radius: 6.5, angle: 0, dist: 0 },
            { id: "sub_1", name: "Intruder Submarine", x: 75, y: 40, type: "Submarine", radius: 4.5, angle: 0, dist: 0 }
        ];

        // Wave ripple overlays and ambient bubble currents
        this.bubbles = [];
        this.ripples = [];
        this.currentAngle = 0;
        this.beamWidth = 6.0;
        
        // Interaction states
        this.draggedTarget = null;
        this.selectedTarget = null;
        this.hoveredTarget = null;
        
        this.resizeCanvas();
        this.initBubbles();
        this.setupEventListeners();
        
        // Run update calculations
        this.updateTargetPolarCoordinates();
    }

    resizeCanvas() {
        // Find parent container dimensions
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    initBubbles() {
        // Spawn ambient bubble particles
        this.bubbles = [];
        for (let i = 0; i < 25; i++) {
            this.bubbles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: Math.random() * 2 + 0.5,
                speedY: -(Math.random() * 0.4 + 0.1),
                speedX: Math.random() * 0.2 - 0.1,
                alpha: Math.random() * 0.4 + 0.1
            });
        }
    }

    setupEventListeners() {
        // Drag-and-drop coordinates mapper
        const getMousePos = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            return {
                x: ((e.clientX - rect.left) / rect.width) * 100,
                y: ((e.clientY - rect.top) / rect.height) * 100
            };
        };

        this.canvas.addEventListener('mousedown', (e) => {
            const pos = getMousePos(e);
            
            // Check if user clicked on any target
            this.draggedTarget = this.findTargetAt(pos.x, pos.y);
            if (this.draggedTarget) {
                this.selectedTarget = this.draggedTarget;
                this.canvas.style.cursor = 'grabbing';
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const pos = getMousePos(e);
            
            if (this.draggedTarget) {
                // Keep target bounded inside the sandbox scan area
                this.draggedTarget.x = Math.max(5, Math.min(95, pos.x));
                this.draggedTarget.y = Math.max(5, Math.min(85, pos.y));
                this.updateTargetPolarCoordinates();
            } else {
                // Change cursor to pointer if hovering over a target node
                this.hoveredTarget = this.findTargetAt(pos.x, pos.y);
                this.canvas.style.cursor = this.hoveredTarget ? 'pointer' : 'default';
            }
        });

        const stopDrag = () => {
            if (this.draggedTarget) {
                this.draggedTarget = null;
                this.canvas.style.cursor = 'default';
            }
        };

        this.canvas.addEventListener('mouseup', stopDrag);
        this.canvas.addEventListener('mouseleave', stopDrag);
        
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.initBubbles();
        });
    }

    findTargetAt(pctX, pctY) {
        // Search through targets for close hits
        for (let target of this.targets) {
            const dx = target.x - pctX;
            // Account for aspect ratio scale distortion (approx 1.8x stretch in width)
            const dy = target.y - pctY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist <= target.radius) {
                return target;
            }
        }
        return null;
    }

    /**
     * Converts raw 2D grid coordinates relative to the bottom center sonar sensor
     * into polar coordinates (distance in meters, angle in degrees from the horizontal).
     */
    updateTargetPolarCoordinates() {
        const gridScaleToMeters = 0.06; // Maps grid bounds to max 6 meters
        
        for (let target of this.targets) {
            const dx = target.x - this.sonarPos.x;
            const dy = this.sonarPos.y - target.y; // Invert Y as canvas Y points down
            
            // Raw Euclidean distance
            const gridDist = Math.sqrt(dx * dx + dy * dy);
            target.dist = gridDist * gridScaleToMeters;
            
            // Angle in degrees from horizontal (0 to 180 left to right)
            let rad = Math.atan2(dy, dx);
            let deg = (rad * 180) / Math.PI;
            
            // Adjust angle orientation (radar 0 deg is left, 180 is right)
            // atan2(dy, dx) returns angle from positive X axis
            target.angle = 180.0 - deg;
        }
    }

    spawnTarget(type) {
        const id = `${type.toLowerCase()}_${Date.now()}`;
        const name = `Virtual ${type} Node`;
        // Spawn target at a random location near the upper-middle sector
        const x = Math.random() * 70 + 15;
        const y = Math.random() * 45 + 15;
        
        this.targets.push({
            id: id,
            name: name,
            x: x,
            y: y,
            type: type,
            radius: type === "Fish" ? 6.5 : type === "Rock" ? 5.5 : 4.5,
            angle: 0,
            dist: 0
        });
        
        this.updateTargetPolarCoordinates();
        
        // Spawn feedback ripple at target center
        this.ripples.push({
            x: x,
            y: y,
            r: 1,
            maxR: 15,
            alpha: 0.8
        });
        
        return id;
    }

    deleteTarget(id) {
        this.targets = this.targets.filter(t => t.id !== id);
        if (this.selectedTarget && this.selectedTarget.id === id) {
            this.selectedTarget = null;
        }
    }

    /**
     * Scans the ocean grid at a specific query angle and reports the closest intersecting target.
     * Returns: { class: str, distance: float }
     */
    getScanResult(queryAngle, beamWidth) {
        let closestTarget = null;
        let minDistance = 999.0;
        
        for (let target of this.targets) {
            const angleDiff = Math.abs(queryAngle - target.angle);
            
            // Intersection occurs if queryAngle falls inside target's angular sweep footprint
            // We approximate this by target's angular width: sin(theta) = R / D => theta = asin(R/D)
            const angularRadiusRad = Math.asin(target.radius / Math.sqrt(Math.pow(target.x - this.sonarPos.x, 2) + Math.pow(this.sonarPos.y - target.y, 2)));
            const angularRadiusDeg = (angularRadiusRad * 180) / Math.PI;
            
            // Take beam width overlap into account
            if (angleDiff <= (angularRadiusDeg + beamWidth / 2.0)) {
                if (target.dist < minDistance && target.dist <= 5.8) {
                    minDistance = target.dist;
                    closestTarget = target;
                }
            }
        }
        
        if (closestTarget) {
            return {
                class: closestTarget.type,
                distance: closestTarget.dist,
                target: closestTarget
            };
        }
        
        return {
            class: "Void",
            distance: -1.0,
            target: null
        };
    }

    /**
     * Main Animation & Render loop for the Sandbox Canvas.
     */
    render(activeAngle, beamWidth) {
        this.currentAngle = activeAngle;
        this.beamWidth = beamWidth;
        
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Clear with slight water gradient overlay
        this.ctx.fillStyle = '#020d1c';
        this.ctx.fillRect(0, 0, w, h);
        
        // 1. Draw Grid Overlay (Bathymetry contour lines)
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.03)';
        this.ctx.lineWidth = 1;
        for (let i = 0; i <= w; i += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(i, 0);
            this.ctx.lineTo(i, h);
            this.ctx.stroke();
        }
        for (let i = 0; i <= h; i += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, i);
            this.ctx.lineTo(w, i);
            this.ctx.stroke();
        }
        
        // 2. Render Bubbles (Acoustic Noise particles)
        this.ctx.fillStyle = 'rgba(0, 240, 255, 0.15)';
        for (let b of this.bubbles) {
            this.ctx.beginPath();
            this.ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Move bubbles upward
            b.y += b.speedY;
            b.x += b.speedX;
            
            // Loop bubbles back at the bottom
            if (b.y < 0) {
                b.y = h;
                b.x = Math.random() * w;
            }
        }
        
        // Convert percentage coordinates to canvas absolute pixels
        const toPx = (pctX, pctY) => {
            return {
                x: (pctX / 100) * w,
                y: (pctY / 100) * h
            };
        };

        const sp = toPx(this.sonarPos.x, this.sonarPos.y);

        // 3. Render the active sweeping Sonar Beam cone
        const radCenter = (180.0 - activeAngle) * Math.PI / 180.0;
        const radLeft = radCenter - (beamWidth / 2.0) * Math.PI / 180.0;
        const radRight = radCenter + (beamWidth / 2.0) * Math.PI / 180.0;
        const scanRange = Math.max(w, h) * 0.85;

        const scanGrad = this.ctx.createRadialGradient(sp.x, sp.y, 10, sp.x, sp.y, scanRange);
        scanGrad.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
        scanGrad.addColorStop(0.3, 'rgba(0, 240, 255, 0.08)');
        scanGrad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

        this.ctx.fillStyle = scanGrad;
        this.ctx.beginPath();
        this.ctx.moveTo(sp.x, sp.y);
        this.ctx.arc(sp.x, sp.y, scanRange, radLeft, radRight, false);
        this.ctx.closePath();
        this.ctx.fill();
        
        // Draw thin vector line down the sweep center
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
        this.ctx.lineWidth = 1.5;
        this.ctx.beginPath();
        this.ctx.moveTo(sp.x, sp.y);
        this.ctx.lineTo(sp.x + scanRange * Math.cos(radCenter), sp.y + scanRange * Math.sin(radCenter));
        this.ctx.stroke();

        // 4. Render Target Nodes
        for (let target of this.targets) {
            const tp = toPx(target.x, target.y);
            const rPx = (target.radius / 100) * w;
            
            // Set styles based on target type
            let color = varToHex('--neon-blue');
            let icon = 'mountain';
            
            if (target.type === "Fish") {
                color = varToHex('--neon-emerald');
                icon = 'fish';
            } else if (target.type === "Submarine") {
                color = varToHex('--neon-red');
                icon = 'ship';
            }
            
            // Highlight if hovered, selected, or actively scanned
            const angleDiff = Math.abs(activeAngle - target.angle);
            const isScanned = angleDiff <= (beamWidth * 1.5);
            
            // Drop target shadow/glow
            this.ctx.shadowBlur = (this.selectedTarget === target) ? 20 : isScanned ? 12 : 0;
            this.ctx.shadowColor = color;

            // Draw outer dotted ring for acoustic cross-section boundary
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = (this.selectedTarget === target) ? 2.5 : 1.2;
            this.ctx.setLineDash(target.type === "Fish" ? [4, 4] : []);
            this.ctx.beginPath();
            this.ctx.arc(tp.x, tp.y, rPx, 0, Math.PI * 2);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            // Core target filler
            const fillGrad = this.ctx.createRadialGradient(tp.x, tp.y, 2, tp.x, tp.y, rPx);
            fillGrad.addColorStop(0, color + '55');
            fillGrad.addColorStop(1, color + '15');
            this.ctx.fillStyle = fillGrad;
            this.ctx.beginPath();
            this.ctx.arc(tp.x, tp.y, rPx - 1, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw center blip dot
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(tp.x, tp.y, 3, 0, Math.PI * 2);
            this.ctx.fill();

            // Reset shadows
            this.ctx.shadowBlur = 0;
            
            // Render text tag labels
            this.ctx.fillStyle = '#fff';
            this.ctx.font = 'bold 9px var(--font-primary)';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(target.name.toUpperCase(), tp.x, tp.y - rPx - 8);
            
            this.ctx.fillStyle = '#8da2bb';
            this.ctx.font = '7px var(--font-mono)';
            this.ctx.fillText(`${target.dist.toFixed(2)}m @ ${Math.round(target.angle)}°`, tp.x, tp.y - rPx - 1);
        }
        
        // 5. Draw the physical sonar transducer dome (emitter)
        this.ctx.fillStyle = '#091c33';
        this.ctx.strokeStyle = varToHex('--neon-cyan');
        this.ctx.lineWidth = 2.5;
        this.ctx.beginPath();
        this.ctx.arc(sp.x, sp.y, 14, Math.PI, 0); // half dome
        this.ctx.fill();
        this.ctx.stroke();
        
        this.ctx.fillStyle = varToHex('--neon-cyan');
        this.ctx.beginPath();
        this.ctx.arc(sp.x, sp.y, 4, 0, Math.PI * 2); // core emitter blip
        this.ctx.fill();
    }
}

// Helper to pull hex values out of CSS variables dynamically
function varToHex(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}
