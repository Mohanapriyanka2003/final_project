/**
 * sonar.js — Canvas sonar renderer
 */
const Sonar = (() => {
  let canvas, ctx, CX, CY, R;

  function init(id) {
    canvas=document.getElementById(id); ctx=canvas.getContext("2d");
    CX=canvas.width/2; CY=canvas.height/2; R=canvas.width/2-6;
  }

  function render(deg, objects) {
    if(!ctx) return;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    _bg(); _rings(); _grid(); _trail(deg); _line(deg); _echoes(objects); _border();
  }

  function _bg(){
    const g=ctx.createRadialGradient(CX,CY,0,CX,CY,R);
    g.addColorStop(0,"#041a2e"); g.addColorStop(1,"#010b16");
    ctx.fillStyle=g; ctx.beginPath(); ctx.arc(CX,CY,R,0,Math.PI*2); ctx.fill();
  }
  function _rings(){
    [.25,.5,.75,1].forEach((f,i)=>{
      ctx.beginPath(); ctx.arc(CX,CY,R*f,0,Math.PI*2);
      ctx.strokeStyle=`rgba(0,180,255,${.07+i*.05})`; ctx.lineWidth=1; ctx.stroke();
      ctx.fillStyle="rgba(0,200,255,.4)"; ctx.font="8px monospace";
      ctx.fillText(`${Math.round(200*f)}cm`,CX+R*f+3,CY-3);
    });
  }
  function _grid(){
    for(let a=0;a<360;a+=30){
      const r=a*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(CX,CY); ctx.lineTo(CX+R*Math.cos(r),CY+R*Math.sin(r));
      ctx.strokeStyle="rgba(0,140,210,.08)"; ctx.lineWidth=.5; ctx.stroke();
    }
  }
  function _trail(deg){
    const sr=deg*Math.PI/180;
    for(let i=0;i<85;i++){
      const a=sr-i*.016;
      ctx.beginPath(); ctx.moveTo(CX,CY); ctx.arc(CX,CY,R,a-.016,a);
      ctx.fillStyle=`rgba(0,255,110,${(85-i)/85*.18})`; ctx.fill();
    }
  }
  function _line(deg){
    const sr=deg*Math.PI/180;
    ctx.beginPath(); ctx.moveTo(CX,CY); ctx.lineTo(CX+R*Math.cos(sr),CY+R*Math.sin(sr));
    ctx.strokeStyle="rgba(0,255,90,.95)"; ctx.lineWidth=2;
    ctx.shadowColor="#00ff60"; ctx.shadowBlur=10; ctx.stroke(); ctx.shadowBlur=0;
  }
  function _echoes(objects){
    objects.forEach(obj=>{
      (obj.echoes||[]).forEach(e=>{
        const ex=CX+(e.r/200)*R*Math.cos(e.a), ey=CY+(e.r/200)*R*Math.sin(e.a);
        const al=e.life/220;
        const g=ctx.createRadialGradient(ex,ey,0,ex,ey,obj.sz+10);
        g.addColorStop(0,obj.color+_h(al*140)); g.addColorStop(1,"transparent");
        ctx.beginPath(); ctx.arc(ex,ey,obj.sz+10,0,Math.PI*2); ctx.fillStyle=g; ctx.fill();
        ctx.beginPath(); ctx.arc(ex,ey,obj.sz,0,Math.PI*2);
        ctx.fillStyle=obj.color+_h(al*255);
        ctx.shadowColor=obj.color; ctx.shadowBlur=12*al; ctx.fill(); ctx.shadowBlur=0;
      });
    });
  }
  function _border(){
    ctx.beginPath(); ctx.arc(CX,CY,R,0,Math.PI*2);
    ctx.strokeStyle="rgba(0,180,255,.28)"; ctx.lineWidth=2; ctx.stroke();
    ctx.beginPath(); ctx.arc(CX,CY,5,0,Math.PI*2);
    ctx.fillStyle="#00e5ff"; ctx.shadowColor="#00e5ff"; ctx.shadowBlur=12; ctx.fill(); ctx.shadowBlur=0;
  }
  function _h(n){ return Math.round(Math.max(0,Math.min(255,n))).toString(16).padStart(2,"0"); }

  return {init, render};
})();
